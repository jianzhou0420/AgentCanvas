"""Place memory — the collapse operator.

The metric map (``depthmap.AnchorMap``) is an INSTRUMENT, not a memory. It is
precise, it drifts, and everything it knows is expressed in metres and in
left/right — two things that do not survive walking away and turning around.
This module is what survives: places, what was seen in each, and the order they
were walked. No metres. No left/right. No step numbers.

The design and the adversarial review that reshaped it are written up at
blackboard docs-site pages/navharness/design/place-memory.html. Three points
from that review are load-bearing here and are marked in the code:

  * a place boundary must come from EGO-MOTION, never from the visible landmark
    set turning over — six turn primitives rotate the whole 90° field of view
    with the body standing still, so a turnover-based boundary would invent a
    new place for a pirouette and none at all for walking down a corridor;
  * recognition is emitted as a QUESTION, never as an assertion. The path from
    "you have been here" to an armed stop condition has no gate in it, and a
    9B cannot recover from a confidently asserted false memory;
  * collapse may produce empty CONTENT but must never skip the NODE — an absent
    node reads downstream as "I have not been here", which is the false positive
    that makes lost-detection useless.

What the model receives is two sentences and nothing else:
    此地：吧台在你右前方，椅子在你左边
    走过：先经过了有泳池的地方，然后到这里
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eharness.depthmap import CELL_M, FREE, TopDown

# Ego-motion boundary thresholds. A doorway pinches the free width and then
# releases it; that pinch-release is the one place event a body can feel without
# any world frame. The displacement fallback is deliberately LARGE so that a
# corridor stays one richly-populated place rather than being chopped into
# arbitrary segments — the review's point that a corridor is a place, not a
# sequence of places.
PINCH_M = 1.6            # free width narrower than this = going through something
RELEASE_M = 2.6          # …and wider than this afterwards = came out the far side
PROBE_AHEAD_M = 2.0      # where across the body the width is measured.
#   NOT closer: a 90° field of view can only see ±d of lateral extent at
#   distance d, so probing at 1 m caps the reading at 2 m and the release
#   threshold below becomes unreachable — every room would read as narrow as
#   a doorway. The probe distance is bounded from below by the optics, not by
#   taste.
SPAN_FALLBACK_M = 9.0    # open-plan fallback: this far from entry is a new place
MIN_PLACE_MOVES = 4      # a place has to be lived in before it can be left


def free_width(td: TopDown, ahead_m: float = PROBE_AHEAD_M) -> float:
    """Metres of contiguous free floor across the body, one metre ahead.

    Deliberately measured ACROSS the heading rather than along it: walking
    through a doorway barely changes what is ahead, and changes what is beside
    you completely."""
    row = int(ahead_m / CELL_M)
    if not 0 <= row < td.n_fwd:
        return 0.0
    line = td.grid[row]
    mid = td.n_lat // 2
    if line[mid] != FREE:
        return 0.0
    lo = mid
    while lo > 0 and line[lo - 1] == FREE:
        lo -= 1
    hi = mid
    while hi < td.n_lat - 1 and line[hi + 1] == FREE:
        hi += 1
    return (hi - lo + 1) * CELL_M


@dataclass
class Place:
    """One region of space, identified by what is in it — never by a number the
    model sees, and never by where it is."""

    index: int
    entry_xy: tuple[float, float]          # instrument frame; never leaves this module
    seen: dict[str, int] = field(default_factory=dict)   # phrase → looks it appeared in
    order: list[str] = field(default_factory=list)       # first-sighting order, here
    moves: int = 0
    left_by: str = ""                      # which ego-motion event ended it

    def note(self, phrases: list[str]) -> None:
        for p in phrases:
            if p not in self.seen:
                self.order.append(p)
            self.seen[p] = self.seen.get(p, 0) + 1

    def name(self, min_looks: int = 2) -> str:
        """A place is named by its contents, because that is the only handle a
        person has. Anonymous ids were cut from the design: they are an
        implementation detail wearing the clothes of a principle."""
        solid = [p for p in self.order if self.seen[p] >= min_looks]
        if not solid:
            solid = list(self.order)
        if not solid:
            return "a place you could not make out"
        if len(solid) == 1:
            return f"the place with the {solid[0]}"
        return "the place with the " + " and ".join(solid[:2])

    def collapsed(self) -> dict:
        """What survives leaving: names and order. No metres, no left/right."""
        return {"index": self.index, "name": self.name(),
                "contents": list(self.order), "left_by": self.left_by}


class PlaceMemory:
    """The persistent tier. Fed one look at a time; emits sentences, not numbers."""

    def __init__(self) -> None:
        self.places: list[Place] = []
        self.events: list[str] = []        # boundary events, for the human readout
        self._pinched = False
        self._recognised: int | None = None
        self._asked: set[int] = set()      # ask about a place once, not every look

    # ── the walk ─────────────────────────────────────────────────────────
    @property
    def current(self) -> Place | None:
        return self.places[-1] if self.places else None

    def observe(self, td: TopDown | None, phrases: list[str],
                pose: tuple[float, float]) -> str:
        """One look. Returns a boundary event name, or "" if we stayed put."""
        if not self.places:
            self.places.append(Place(index=1, entry_xy=pose))
            self.events.append("起点")

        # Boundary FIRST, contents second. The look that shows you are out of a
        # room is already showing the next room — crediting its landmarks to the
        # place you just left puts the bar in the pool room.
        event = self._boundary(td, pose, self.places[-1])
        if event:
            self.places[-1].left_by = event
            self.places.append(Place(index=self.places[-1].index + 1,
                                     entry_xy=pose))
            self.events.append(event)

        place = self.places[-1]
        place.moves += 1
        place.note(phrases)

        # Recognition is checked once the place has some substance, NOT at the
        # instant of entry — a place one look old is empty, and an empty
        # signature matches nothing, which is how this silently never fired.
        if self._recognised is None and place.index not in self._asked:
            hit = self._match(place)
            if hit is not None:
                self._recognised = hit
                self._asked.add(place.index)
        return event

    def _boundary(self, td: TopDown | None, pose: tuple[float, float],
                  place: Place) -> str:
        if place.moves < MIN_PLACE_MOVES:
            return ""
        if td is not None:
            w = free_width(td)
            if w and w < PINCH_M:
                self._pinched = True
            elif self._pinched and w > RELEASE_M:
                self._pinched = False
                return "穿过了一个窄口"
        dx = pose[0] - place.entry_xy[0]
        dy = pose[1] - place.entry_xy[1]
        if math.hypot(dx, dy) > SPAN_FALLBACK_M:
            return "走出了很远"
        return ""

    def _match(self, place: Place, need: int = 2) -> int | None:
        """Have we been somewhere like this before? Conservative on purpose: a
        FALSE recognition sends the agent confidently the wrong way, which is
        worse than no recognition at all. Requires `need` shared contents and
        never matches the place we just left."""
        here = set(place.seen)
        if len(here) < need:
            return None
        for old in self.places[:-2]:
            if len(here & set(old.seen)) >= need:
                return old.index
        return None

    # ── what the model is told ───────────────────────────────────────────
    def here_sentence(self, sightings: list) -> str:
        """Inside a place, side is still meaningful — but the metres are not
        kept, because they are the part that stops being true the moment the
        body moves.

        ONE entry per phrase: the detector returns up to three instances of each
        noun, and un-deduplicated they render as "chairs on your right, chairs on
        your right, chairs ahead", which reads as three separate sets of chairs.

        English, deliberately: everything else the model is shown is English, and
        switching language mid-prompt is a needless risk for a 9B. The Chinese
        readout is for the human watching the run (`readout`)."""
        if not sightings:
            return ""
        best: dict[str, Any] = {}
        for s in sightings:
            keep = best.get(s.phrase)
            if keep is None or abs(s.bearing) < abs(keep.bearing):
                best[s.phrase] = s
        parts = []
        for s in list(best.values())[:4]:
            deg = math.degrees(s.bearing)
            if abs(deg) < 15:
                where = "straight ahead"
            elif abs(deg) > 70:
                where = "to your " + ("left" if deg > 0 else "right")
            else:
                where = "ahead on your " + ("left" if deg > 0 else "right")
            parts.append(f"the {s.phrase} {where}")
        return "Here: " + ", ".join(parts) + "."

    def walked_sentence(self) -> str:
        """The route, as order. No step numbers — the review's point that a
        person remembers having passed the pool, not the index of the frame."""
        done = [p for p in self.places[:-1] if p.order]
        if not done:
            return ""
        chain = ", then ".join(p.name() for p in done)
        return f"You walked through {chain}, and then to where you are now."

    def recognition_question(self) -> str:
        """Always a question. Never an assertion, and it must not touch any
        stop condition — the path from `at_goal` to an armed brake has no gate
        in it, so an asserted false memory is unrecoverable."""
        if self._recognised is None:
            return ""
        old = next((p for p in self.places if p.index == self._recognised), None)
        self._recognised = None
        if old is None:
            return ""
        return (f"This looks like {old.name()}, which you left earlier — is it "
                "the same place? If it is, you have come back round.")

    def lines(self, sightings: list) -> list[str]:
        return [s for s in (self.here_sentence(sightings),
                            self.walked_sentence(),
                            self.recognition_question()) if s]

    # ── what a human is shown ────────────────────────────────────────────
    def dump(self) -> dict:
        return {
            "places": [p.collapsed() | {"looks": p.moves,
                                        "counts": dict(p.seen)} for p in self.places],
            "events": list(self.events),
            "current": self.current.name() if self.current else "",
        }

    def readout(self) -> str:
        rows = []
        for p in self.places:
            mark = "  ← 现在" if p is self.current else ""
            body = "、".join(f"{k}×{v}" for k, v in p.seen.items()) or "（还什么都没认出来）"
            rows.append(f"  地点{p.index} {p.name()}：{body}"
                        + (f"  [离开：{p.left_by}]" if p.left_by else "") + mark)
        return "\n".join(rows) or "  （还没有任何地点）"

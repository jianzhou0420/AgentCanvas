---
name: nav-instruction-refiner
description: "Clean and conservatively clarify a navigation instruction before an EmbodiedHarness episode. Use only the instruction text: remove non-spatial filler, smooth disfluency, and clarify a reference only when the wording itself forces the referent. Never use scene assumptions or visual grounding."
---

# Navigation Instruction Refiner

Clean a natural-language navigation instruction for downstream navigation. Perform only these operations:

1. Remove filler that carries no spatial, landmark, action, or stopping information.
2. Smooth disfluency and awkward wording without changing meaning.
3. Clarify a reference only when the instruction's own action order or spatial wording forces the referent.

Use no visual or scene information. Do not act as a disambiguation decider or ask a clarification question. Preserve anything that cannot be resolved from the text alone.

## Refine conservatively

- Preserve the input language. Never translate the instruction or its landmarks.
- Preserve sentence boundaries and action order. Polish within sentences; do not merge separate sentences or split one sentence into several.
- Remove a clause only when it has no spatial, landmark, action, verification, or endpoint value. Keep apparent commentary such as "You will pass a painting on your left" because it is a route-verification cue.
- Clarify only when the wording forces the answer through traversal entailment, explicit ordering, a stated egocentric direction, or a stated landmark anchor.
- Never infer the environment's layout, number of doors, nearer side, furniture shape, or any other unseen fact.
- Never invent a referent or landmark, delete spatial content, change route granularity, reorder actions, or alter the endpoint.
- Return non-navigation input unchanged with `clarified: false`.

Before answering, check that the result describes the same route and endpoint and has lost no spatial content. If uncertain, retain the original wording.

## Output contract

Return exactly one JSON object and nothing else. Do not add Markdown fences, analysis, or surrounding prose.

```json
{"instruction":"<cleaned text>","clarified":false,"note":"<optional audit-only explanation>"}
```

- `instruction` is required and must be a non-empty string.
- `clarified` is required and is true only when a text-derivable reference was made explicit.
- `note` is optional, audit-only, and never enters the navigation prompt.

## Examples

Input:

`Go straight past the pool. Walk between the bar and chairs. Stop when you get to the corner of the bar. That's where you will wait.`

Output:

```json
{"instruction":"Go straight past the pool. Walk between the bar and the chairs. Stop when you reach the far corner of the bar.","clarified":true,"note":"The prior between-traversal makes the far corner the text-entailed stopping point."}
```

The traversal wording entails passing through the gap before stopping, so `far corner` is text-derivable. The final sentence is removable because the preceding sentence already contains the complete endpoint. Keep all other action order and sentence boundaries unchanged.

Input:

`Turn right at the hallway. Wait by the door.`

Output:

```json
{"instruction":"Turn right at the hallway. Wait by the door.","clarified":false}
```

The text does not identify which door; preserve it.

Input:

`okay so you wanna like go up the stairs, and then uh the room at the end, wait there.`

Output:

```json
{"instruction":"Go up the stairs, then wait in the room at the end.","clarified":false}
```

Input:

`Walk down the corridor. You'll pass a painting on your left. Stop at the double doors.`

Output:

```json
{"instruction":"Walk down the corridor. You will pass a painting on your left. Stop at the double doors.","clarified":false}
```

Keep the painting clause because it supplies both a landmark and a direction.

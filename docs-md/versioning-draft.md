# Versioning Policy — DRAFT

> **Status: draft (2026-06-28).** Proposed replacement for [`VERSIONING.md`](VERSIONING.md).
> Key change vs the current policy: the **version number is decoupled from the paper**.
> `v1.0.0` is triggered by API stability, not by Paper #1; academic citation is handled
> on a separate axis (Zenodo DOI + `CITATION.cff`). See the *What changed* delta at the end.

This is the engineering-level versioning **contract** for AgentCanvas — how releases are
numbered, tagged, cut, recorded, and cited. The *paradigm-level* meaning of each major
version (what v1 vs v2 actually *covers*) lives in
[`major-versions.html`](../docs/pages/developer-guide/core/major-versions.html); this file
says how those versions are mechanically expressed in git and on the forge.

---

## Three separate concerns — do not conflate

Mature projects keep these on independent axes. Forcing one to track another (e.g.
"`v1.0` = paper is out") couples things that have no reason to move together.

| Concern | Question it answers | Where it lives |
|---|---|---|
| **Policy** (this file) | How do we number, tag, and cut releases? | `versioning-draft.md` → `VERSIONING.md` |
| **Record** | What changed in each version? | `CHANGELOG.md` (Keep a Changelog) + each GitHub Release body |
| **Citation** | How does a paper cite a fixed snapshot? | `CITATION.cff` + Zenodo DOI |

The **version number** is driven by exactly one thing: **the state of the public API**.
It is *not* driven by publication milestones, marketing, or calendar.

---

## TL;DR

- Versions follow **[SemVer 2.0](https://semver.org/)** — `vMAJOR.MINOR.PATCH`, leading `v`.
- All tags are **annotated**, cut from `master` only, and **append-only** (never `-f`).
- **v0.x** is pre-stable: the API may break between *any* two versions, including patches.
- **v1.0.0 ships when the public API is declared stable** (criteria below) — independent of
  any paper. From v1.0.0 onward, strict SemVer backward-compatibility binds.
- **v2.0.0** ships when the executor supports **runtime topology growth**
  (see `major-versions.html` §2). This paradigm boundary is the only thing that justifies a
  MAJOR bump today.
- Pre-releases use `-rc.N` only (no `-alpha` / `-beta`); they mean *feature-complete,
  fixing only release-blocking bugs*.
- **Citation is a separate axis**: when a paper is submitted, tag the *then-current* version
  (typically a `v0.x` or `-rc`), let Zenodo mint a DOI, and cite that DOI. The paper never
  forces the version number.
- Every tag gets a **GitHub Release** (changelog + optional assets). The tag is ground truth;
  the Release is human-facing packaging.

---

## Version number format

```
v MAJOR . MINOR . PATCH [ - rc . N ]
^                       ^
mandatory 'v' prefix    only -rc.N pre-release suffix is used
```

**Valid:** `v0.1.0` · `v0.7.3` · `v1.0.0-rc.1` · `v1.0.0` · `v1.4.2` · `v2.0.0-rc.1`
**Invalid:** `0.1.0` (no `v`) · `v1.0` (no patch) · `v1.0.0-alpha.1` / `-beta.1` (only `-rc`) · `v1.0.0+build.42` (no build metadata)

The `v` prefix is a git convention (Linux kernel, Kubernetes, Go, Node.js). Package
managers (pip, npm, cargo) strip it on consumption, so it does **not** leak into
`pyproject.toml` / `package.json` `version:` fields — those carry the bare `0.1.0`.

**Single source of truth for the version string:** the git tag is authoritative. The
`version` field in `pyproject.toml` (and `package.json`, if published) is bumped in the same
commit that precedes the tag, and must match the tag minus its `v` prefix. No version
string is hardcoded anywhere else.

---

## SemVer bump rules

| Bump | When | Examples |
|---|---|---|
| **MAJOR** | Paradigm shift across the v1 ↔ v2 boundary in `major-versions.html` — today this means introducing **runtime topology mutation**. (Post-1.0, also any backward-incompatible public-API break.) | v1 → v2 |
| **MINOR** | Backward-compatible new capability: new nodeset, new env/method port, new framework feature (F1–F7), deployable/packaging polish, eval-stack additions. | v1.1, v1.2, … |
| **PATCH** | Backward-compatible bug fix or non-functional cleanup. No new user-facing capability. | v1.4.1, v1.4.2, … |

**v1.x vs v2 routing rule of thumb:** if a feature can be expressed with bounded,
predeclared topology — even via router-and-bounded-pool, hidden-in-node, or `K_max`
over-provisioning — it is **v1.x**. If it can exist *only* with runtime topology mutation,
it is **v2**. Any roadmap item marked "moved to v2" in `roadmap.html` stays out of the v1.x
minor stream.

---

## What the public API is

SemVer protects the **public API**. For AgentCanvas that surface is explicitly:

1. **Graph JSON schema** — the on-disk format of a saved agent (node/wire/port/state/config
   shape). A graph that loads on `vX.Y` must keep loading on every later `vX.*`.
2. **Nodeset author contracts** — the `BaseCanvasNode` / `BaseNodeSet` base-class APIs that
   external nodeset authors subclass.
3. **CLI / HTTP surfaces** — documented commands and backend endpoints used to run/eval graphs.

Internal implementation (executor internals, private helpers, frontend wiring) is **not**
part of the contract and may change in any release. When in doubt, a symbol is public only
if it is documented as such.

---

## v0.x — the pre-stable phase

v0.x follows SemVer's v0 interpretation: **no backward-compatibility guarantee** between any
two v0 versions, including patch bumps. External consumers should treat any `v0.x` pin as
snapshot-only and expect to adapt when bumping.

In practice during v0.x:
- We still bump MINOR for new capability and PATCH for fixes (for human readability), but the
  SemVer compatibility contract does not yet bind.
- Breaking changes do not require a MAJOR bump and do not require a deprecation window — but
  they **must** be recorded in `CHANGELOG.md` under *Changed* / *Removed*.
- API churn from research iteration is expected and accepted here — which is precisely why we
  stay in v0.x until the surfaces above settle.

---

## v1.0.0 — meaning and ship criteria

**`v1.0.0` means: the public API is frozen and we now commit to strict SemVer.** It is a
statement about the *software*, not about a publication.

v1.0.0 ships when **all** of the following hold:

1. **Open-sourceable** — public repository with `LICENSE`, install docs, and `CONTRIBUTING`.
2. **Coverage** — the v1 "Agent forms covered" list in `major-versions.html` §1 is each
   exercised by at least one canonical, runnable graph.
3. **API declared stable** — the three public surfaces above are documented and frozen under
   SemVer; MINOR/PATCH bumps from here must remain backward-compatible.

> **Note (decoupling).** Paper submission is deliberately **not** a v1.0 criterion. A stable,
> open-sourced codebase can reach 1.0 whether or not a paper is in review; conversely, a paper
> may be submitted while the code is still legitimately `v0.x`. Tying the SemVer major to an
> external review timeline would make the version number — whose entire job is to predict
> compatibility — hostage to events unrelated to compatibility. The paper is handled on the
> citation axis below.

---

## Release candidates

The only pre-release suffix is `-rc.N`, and it carries a precise meaning: **feature-complete
for the target version, accepting only release-blocking bug fixes.**

| When | Tag | Trigger |
|---|---|---|
| Target version is feature-complete | `vX.0.0-rc.1` | Code frozen except for fixes to release-blocking bugs |
| Each subsequent stabilization round | `vX.0.0-rc.2`, … | A round of fixes lands; re-cut for testing |
| No release-blocking bugs remain | `vX.0.0` | Final release |

Ordering follows SemVer precedence: `v1.0.0-rc.1 < v1.0.0-rc.2 < v1.0.0`. We do **not** use
`-alpha` / `-beta`: `v0.x` already covers "early / unstable", and `-rc` is reserved for the
final stabilization window.

---

## Citation & archival — the independent axis

Academic citation is decoupled from the version number. Mechanism:

- **`CITATION.cff`** at the repo root. GitHub natively renders a *"Cite this repository"*
  button from it and exports BibTeX/APA. It carries authors, title, and (once minted) the DOI.
- **Zenodo ↔ GitHub integration.** Enabling it makes Zenodo mint a DOI for **every** GitHub
  Release automatically: one *version DOI* per release plus one *concept DOI* that always
  resolves to the latest version. This is the standard archival path for research software.

**When a paper is submitted or revised:**

1. Tag the **then-current** version — whatever the code state warrants (almost always a
   `v0.x` or a `-rc`), **not** a forced `v1.0.0`.
2. Cut the matching GitHub Release; Zenodo mints the version DOI.
3. Cite that **version DOI** in the paper, and record the citation in `CITATION.cff`.

This gives the paper a permanent, reproducible snapshot to point at while keeping the version
number an honest statement about API stability.

---

## Tag mechanics

**All tags are annotated** (never lightweight — lightweight tags lose tagger/date/message
that `git describe` and the Release UI rely on):

```bash
git tag -a v0.1.0 -m "v0.1.0 — first tagged snapshot"
git push origin v0.1.0
```

- **Cut from `master` only.** No tags on feature branches or stale commits. To patch a
  released version, commit the fix on `master`, then tag — we do not run parallel hotfix
  branches off old tags.
- **Append-only.** Never `git tag -f`, never force-push a tag. A mistaken tag is left in place
  and its deprecation announced in the next release's notes; rewriting history breaks anyone
  who pinned the old hash.

---

## Tag vs Release

Two distinct concepts:

| | Tag | Release |
|---|---|---|
| Owner | git itself | GitHub (the forge) |
| What it is | Immutable pointer to a commit hash | Metadata wrapper: changelog text, assets, pre-release flag, DOI hook |
| Mandatory? | **Yes** — the tag is ground truth | Recommended (and required for Zenodo DOI minting) |
| Editable later? | No (neither pointer nor `-m`) | Notes and assets, yes; the tag binding stays immutable |

For each tagged version, create a GitHub Release with:
- **Title:** same as the tag (e.g. `v0.1.0`).
- **Body:** changelog since the previous release (Added / Changed / Fixed / Removed).
- **Pre-release flag:** checked for every `-rc.N` tag **and** every `v0.x` tag.
- **Assets:** attach only if a build artifact exists (Docker ref, wheel); source-only releases
  rely on GitHub's auto-generated tarballs.

---

## CHANGELOG

A top-level `CHANGELOG.md` is the human-readable version **record**, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format: one section per tagged
version, newest first, grouped Added / Changed / Fixed / Removed / Security. An `[Unreleased]`
section accrues entries between tags and is renamed to the version at tag time.

Until `CHANGELOG.md` exists, each GitHub Release body serves as the per-version changelog.
Generation stays **hand-written** until cadence exceeds ~1 release/month (then revisit
`git-cliff` / `release-please`).

---

## Deprecation policy

Within a stable major (≥ v1), capability is removed in two steps, never one:

1. **Deprecate** — mark the symbol/field/endpoint deprecated in code (runtime warning where
   feasible) and in `CHANGELOG.md` under *Changed*, naming the replacement and the earliest
   version that may remove it. Deprecation lands in a MINOR; the feature keeps working.
2. **Remove** — only at the next **MAJOR**.

During v0.x there is no mandatory deprecation window (removals may be immediate), but every
removal is still recorded under *Removed*.

---

## Security & supply chain

- **`SECURITY.md`** declares the supported versions and the private disclosure channel. As a
  pre-1.0 research project, only the latest `v0.x` / `master` is supported; this expands at 1.0.
- **Signed tags** (GPG or Sigstore) are deferred until the first external contributor lands,
  then applied to all release tags going forward.
- **Branch protection on `master`** (no force-push, require PR + CI) is configured at
  open-source-release time.

---

## Cheatsheet

Cut a new tag:
```bash
git checkout master && git pull
# bump version in pyproject.toml / package.json to match (bare, no 'v'), commit
git tag -a v0.X.Y -m "v0.X.Y — short one-line summary"
git push origin master v0.X.Y
gh release create v0.X.Y --title v0.X.Y --notes-file - < changelog-snippet.md   # mints Zenodo DOI
```

List / inspect tags:
```bash
git tag -l 'v*' --sort=-v:refname
git tag -n99 v0.X.Y
```

---

## Open / deferred decisions

- **CHANGELOG generation** — hand-written vs `git-cliff` vs `release-please`. Hand-written
  until cadence > ~1/month.
- **Tag signing** — GPG vs Sigstore. Defer until first external contributor.
- **DOI granularity** — version DOI per release is automatic; decide whether to advertise the
  concept DOI (always-latest) in `README` / `CITATION.cff`.
- **Backport policy** for patch fixes to older majors (e.g. v1.4.x after v2.0.0). Defer until
  v2 work begins.

---

## What changed vs the current `VERSIONING.md`

For review — this draft differs from `VERSIONING.md` (2026-05-18) in exactly these ways:

1. **v1.0.0 decoupled from the paper.** Ship criteria become *open-sourceable + v1 coverage +
   public-API-declared-stable*. The old criterion "(2) Paper #1 submitted" is removed.
2. **`-rc.N` redefined.** Was pegged to "Paper #1 submission / revision / accepted"; now means
   the generic *feature-complete, bugfix-only* stabilization window.
3. **New "Citation & archival" axis.** `CITATION.cff` + Zenodo DOI, explicitly tagged at the
   *then-current* version rather than at a forced `v1.0.0-rc.1`.
4. **New "What the public API is" section** — names the three contract surfaces SemVer
   actually protects (graph JSON schema, nodeset base-class contracts, CLI/HTTP).
5. **New "Three separate concerns" framing** (policy / record / citation) and a
   **single-source-of-truth** rule for the version string.
6. **Added Deprecation policy and Security & supply chain sections.**
7. Everything else (SemVer format, `v` prefix, v0.x semantics, the v1↔v2 runtime-topology
   boundary, annotated/master-only/append-only tag mechanics, tag-vs-release, CHANGELOG
   format) is **carried over unchanged** from the current policy.

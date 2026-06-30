# NodeSet Layout & Naming

Rules for the directory hierarchy and file naming under `workspace/nodesets/`.
This standard governs **where a nodeset file lives and what it is called**;
how to *write* one is covered by the tutorials
(`.claude/tutorials/skill-nodeset.md`, `skill-env-nodeset.md`) and
`.claude/standard/component-creation.md`.

Adopted 2026-06-12 (role-directory scheme, replacing the deployment-split
`server/` convention). The tree may lag the standard until the TODO #40
migration executes — see "Migration map" at the bottom; the standard is the
target, pre-migration paths are not precedent.

## Core principle

The directory encodes **role** (what kind of thing a nodeset is). Deployment —
which interpreter runs it — is **not** a tree concern: it is fully expressed by
the `server_python` ClassVar (ADR-server-001 / ADR-020 auto-routing), and the
engine reads only that. There is no `server/` directory.

Why role won the directory axis:

- The tree's two readers — humans and the coding agents of Paper #1 (AAS
  implementer) and Paper #2 (porting agents) — both query by role: "which envs
  exist", "where are smartway's parts". Paths are retrieval keys for agents;
  `env/` answers in one hop.
- The doc-site nodeset catalog is already role-organized; one taxonomy
  everywhere, not one for docs and another for code.
- Grouping by interpreter scattered method families across the tree
  (`smartway` once lived in four places). Within a role directory,
  family-prefix sorting makes the parts physically adjacent for free.
- The scanner (`WorkspaceComponentRegistry._scan_subdir`) supports any number of
  top-level buckets natively — this scheme costs zero engine changes.

## Directory grammar

```
workspace/nodesets/
├── env/          # simulators / benchmark environments
├── method/       # paper systems: reasoning core + that method's owned parts
├── policy/       # trained-policy inference wrappers
├── model/        # generic foundation / perception model servers
├── common/       # deliberately cross-method utility tools
├── other/        # quarantine: not yet classified (see rules below)
└── _upstream/    # frozen vendored reference code (scanner-invisible)
```

Inside every role directory, a nodeset is either a single file or a folder
package:

```
{role}/{name}.py               # single file — the default
{role}/{name}/__init__.py      # folder package — once any sidecar exists
```

1. **Role directories are buckets** — no `__init__.py` in them, ever (that
   would flip the scanner into package mode and load the whole directory as
   one nodeset).
2. **Nesting ceiling**: the entry module sits at `{role}/{name}.py` or
   `{role}/{name}/__init__.py` — the scanner descends no further. *Inside* a
   folder package, organize freely (`policy/policy_vla/adapters/…`).
3. **The root holds only the seven directories above.** No loose `.py` at
   root.
4. **No new role directories without an ADR** — each bucket is a top-level
   taxonomy decision, not a tidiness move.
5. **Scanner visibility is opt-out by underscore**: `_`-prefixed files and
   directories are skipped at every level. Everything visible to the scanner
   must be a nodeset entry module — loose helpers, renderers, or tests beside
   nodesets are forbidden; they belong inside their owner's folder package.
   One exception: a `_`-prefixed module shared by **two or more** nodesets in
   the same role directory may sit beside them (e.g.
   `method/_explore_eqa_tsdf.py`, imported by `explore_eqa` and
   `tooleqa_explore`) — single-owner helpers still go inside the owner.
6. **`_upstream/{repo}/`** holds frozen vendored reference code only. Runtime
   code never imports from it (workspace-standalone rule); anything needed at
   runtime is *copied* into the owning nodeset's folder.

### Role definitions

| Directory | Holds | Membership test |
|---|---|---|
| `env/` | simulator / benchmark environments | implements the gym verb contract (`reset` / `step_*` / `observe_*` / `close`) + an `EnvPanel` |
| `policy/` | trained-policy inference | loads checkpoints, maps obs→action; no reasoning prompts |
| `model/` | generic model servers (VLM, tagger, detector, segmenter) | a second, unrelated method could load it **unchanged** and benefit |
| `method/` | one paper/system's reasoning core **and its owned parts** | everything tuned to one method — including its model servers (ownership beats modality: `opennav_perception` lives here, not in `model/`) |
| `common/` | deliberately cross-method utility tools | general by design, not by accident (`basic_agent`, `example`) |
| `other/` | **quarantine** for nodesets that genuinely fit no role yet | last resort, see below |

### `other/` rules

`other/` exists so an unclassifiable nodeset never forces a bad classification
or a root-level dump. It is a holding pen, not a home:

- Entry requires a one-line comment in the module docstring saying *why* it
  doesn't fit the five roles.
- All naming rules still apply inside — no grab-bag names; "I don't know its
  role" never means "I don't know its subject".
- Review `other/` whenever this standard or the role set is touched; moving
  out is a plain `git mv` (deployment never moves, so relocation is free).
- If `other/` accumulates several nodesets sharing a theme, that theme is a
  candidate new role directory → write the ADR.

## Single file vs folder package

- **Single file is the default.** Upgrade to a folder the moment the nodeset
  acquires *any* sidecar: a simulator wrapper, vendored model code, prompt
  files, preset configs, a test.
- **Everything the nodeset owns lives inside its folder** — wrappers
  (`_wrapper.py`), vendored trees, `prompts/`, `presets/`, `test_*.py`.
  Package internals other than `__init__.py` are not auto-imported, so tests
  and helpers are safe there without underscore gymnastics.
- A single-file nodeset that needs a test upgrades to folder form; a loose
  `test_{name}_*.py` sibling is not an option (rule 5 above).

## Naming

1. **Stem = `name` ClassVar, exactly — full prefix included.**
   `env/env_habitat.py` ⇔ `name = "env_habitat"`. The directory + prefix
   redundancy is deliberate: the registry and `node_type` namespace are flat
   (`env_habitat__step`), so the name must be globally self-identifying — in
   an editor tab, a grep hit, or an AAS diff, with no directory in sight.
   When the name carries an abbreviation, the stem follows the **name**
   (`env_mp3d.py`, never `matterport3d.py`).
2. **Role prefixes**:

   | Prefix | Used in | Example |
   |---|---|---|
   | `env_{sim}` | `env/` | `env_libero` |
   | `policy_{name}` | `policy/` | `policy_cma` |
   | `vlm_{name}` | `model/` (vision-language models) | `vlm_prismatic` |
   | `model_{name}` | `model/` (other generic models) | `model_sam`, `model_ram` |
   | *(bare)* `{method}` | `method/` core | `navgpt`, `voxposer` |
   | `{method}_{part}` | `method/` satellites | `smartway_waypoint`, `tooleqa_explore` |
   | descriptive noun | `common/`, `other/` | `basic_agent` |

3. **Method families**: the core takes the bare `{method}` name; every
   satellite is `{method}_{part}`. The shared prefix is mandatory — combined
   with same-directory alphabetical sorting it keeps the family physically
   adjacent.
4. **Method-owned vs generic model** (the TODO #56 boundary, operational):
   would a second unrelated method load it unchanged and benefit? Yes →
   `model/` with `vlm_`/`model_` prefix. No (prompts, label space, or
   preprocessing tuned to one method) → `method/` as `{method}_{part}`.
5. **Node types**: `{nodeset_name}__{verb}_{noun}` (double underscore). Env
   nodesets draw verbs from the gym contract; see `skill-env-nodeset.md`.
   Renaming a nodeset therefore renames its node_types — which graphs bind to.
   File *moves* never touch graphs; *name* changes require a graph-JSON sweep.
6. **Class names**: `XxxNodeSet`, node classes `XxxTool` / `XxxNode`, panel
   classes `XxxEnvPanel` — per `.claude/standard/code-style.md`.
7. **Forbidden names**: grab-bags (`others`, `misc`, `utils`, `common`) — name
   by what the tools *are*; if a set has no one subject, it is several
   nodesets. `common/example.py` is the one blessed exception to "general by
   design" naming (the canonical skeleton, kept import-runnable on purpose).

## Decision table for a new file

| You are adding… | It goes… |
|---|---|
| a simulator / benchmark env | `env/env_{sim}.py` (or `env/env_{sim}/`) + `server_python` |
| a method's reasoning nodeset | `method/{method}.py` |
| a method's satellite (its own env/interpreter is fine) | `method/{method}_{part}.py` + `server_python` as needed |
| a trained-policy wrapper | `policy/policy_{name}…` |
| a generic model server | `model/vlm_{name}.py` or `model/model_{name}.py` |
| a deliberately cross-method tool set | `common/{descriptive_name}.py` |
| something that truly fits none of the above | `other/{descriptive_name}.py` + docstring note why |
| a wrapper / vendored code / prompt / preset / test for nodeset X | inside `X/` (upgrade X to folder form first if needed) |
| upstream repo source for reference | `_upstream/{repo}/` (never imported) |
| a new role directory | nowhere — write an ADR first |

## Migration map (TODO #40)

Target locations for every current nodeset. Two tiers:

**Tier 1 — pure `git mv` (graphs bind `node_type`, not paths; zero graph
impact).** Verify cross-nodeset `from … import` and `.claude`/docs path
references while moving:

| Current | Target |
|---|---|
| `server/habitat.py` / `hmeqa.py` / `matterport3d.py` / `openeqa.py` | `env/env_habitat.py` / `env_hmeqa/` (absorbing `hmeqa_renderer.py`, `hmeqa_replay.py`) / `env_mp3d/` (absorbing `test_matterport3d_path_resolution.py`) / `env_openeqa_em.py` |
| `server/libero/`, `server/simpler/`, `server/env_detany3d/` | `env/env_libero/`, `env/env_simpler/`, `env/env_detany3d/` |
| `server/policy_cma.py`, `policy_octo.py`, `policy_vla/`, `policy_vlnce/` | `policy/…` (names unchanged) |
| `server/vlm_prismatic.py`, `vlm_qwen2_5_vl.py` | `model/…` (names unchanged) |
| `navgpt.py`, `navgpt_mp3d_tools.py`, `mapgpt.py`, `discussnav.py`, `opennav.py`, `spatialnav.py`, `ssg.py` | `method/…` (names unchanged) |
| `smartway/`, `smartway_mono/`, `server/smartway_perception/`, `server/smartway_waypoint/` | `method/smartway*` — family reunited |
| `tooleqa/`, `server/tooleqa_explore.py` | `method/tooleqa*` |
| `server/opennav_perception.py`, `server/opennav_waypoint/` | `method/opennav_*` |
| `server/explore_eqa.py` + `server/_explore_eqa_tsdf.py` | `method/explore_eqa.py` + `method/_explore_eqa_tsdf.py` (tsdf is imported by both `explore_eqa` and `tooleqa_explore`, so it stays a shared `_` module rather than being absorbed by one owner) |
| `voxposer/` | `method/voxposer/` |
| `basic_agent.py`, `example.py` | `common/…` |
| `_upstream/` | stays at root |

**Tier 2 — nodeset renames (node_types change → graph sweep required).**
Known graph blast radius as of 2026-06-12: only
`vln/unverified/discussnav_mp3d.json` (1 node, `ram_perception__tag_panorama`);
`sam__*` and `others__*` appear in no graph:

| Current | Target | Graph impact |
|---|---|---|
| `sam.py` (`sam`) | `model/model_sam.py` (`model_sam`) | none |
| `server/ram_perception.py` (`ram_perception`) | `model/model_ram.py` (`model_ram`) | 1 node in discussnav_mp3d.json |
| `others.py` (`others`) | split by subject into `common/…` | none |

**Known caveat — architect overlays**: `--workspace` overlays resolve by
relative path; historical `design_runs/*/active_workspace/` trees use
pre-migration paths, so replaying old AAS iters after migration will miss
overlays. New runs are unaffected. Migrate at a moment when old-iter replay
is not needed, or add a path-mapping shim first.

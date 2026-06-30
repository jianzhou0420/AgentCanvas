# Contributing to AgentCanvas

last updated: 2026-06-29

Two kinds of contribution, both welcome:

- **Content** — grow the ecosystem: write a nodeset, or compose a graph out of nodesets.
- **Core** — improve the framework itself: UI, backend, features, refactors.

**AgentCanvas** is a [visual agent-design platform for embodied AI research](docs/pages/developer-guide/core/blueprint.html) — you prototype agents by drawing node graphs that run in real time against simulators (Habitat-Sim, MatterSim, SAPIEN/ManiSkill2, MuJoCo/robosuite) or real hardware. For version numbering, tagging, and release flow, see the [Versioning Policy](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html).

---

## 1. Content — nodesets & graphs (most contributions)

This is where most research lands: you extend what AgentCanvas can *do*, without touching the framework.

**Write a nodeset** — a self-contained capability others can drop into a graph. Two flavors:

- **Tool nodesets** turn a capability into nodes — e.g. a real-time 3D Gaussian-Splatting reconstructor, a voxel-based SLAM system, an object detector, or a new simulator.
- **Method nodesets** encode an agent's reasoning — e.g. a NavGPT nodeset, a MapGPT nodeset.

**Or compose a graph** — a JSON that wires existing nodesets into a complete agent method. Often no new Python: the contribution is the topology that turns a pile of nodes into a strong graph agent.

**How**: open a PR adding your files under `workspace/` (`nodesets/`, `nodes/`, or `graphs/`). Review is light — it loads, follows the [component conventions](.claude/standard/component-creation.md), and stays inside `workspace/`.

**You keep the credit.** Every nodeset and graph is attributed to its author/maintainer on the **Credits** board — maintained in the README and a dedicated doc-site page — so contributing here, rather than in a separate repo, doesn't cost you authorship. If your contribution has an associated paper, its citation link goes on the Credits board too.

Start from [`workspace/nodesets/example.py`](workspace/nodesets/example.py) for a nodeset, or an existing file under [`workspace/graphs/`](workspace/graphs/) for a graph. The [nodeset docs](docs/pages/developer-guide/nodesets/index.html) catalog what already exists.

---

## 2. Core — UI, backend, framework (also welcome)

Suggestions, new features, and even refactors of the core (`agentcanvas/backend/app/`, `agentcanvas/frontend/`, `docs/`) are all welcome — the core isn't closed.

**One ask**: if a change is big enough to cost real time, **open a [GitHub Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions) first** and talk it through before you build. A few minutes aligning up front beats sinking days into something that turns out not to fit. Small fixes and self-contained features can go straight to a PR.

Two hard constraints any core PR must respect:

- **Import boundary**: `agentcanvas/backend/app/` never imports domain libraries (`habitat`, `habitat_sim`, `vlnce_baselines`, `habitat_baselines`) — those live in `workspace/` (enforced by `agentcanvas/backend/app/test_import_boundary.py`).
- **Contract changes need an ADR**: touching `BaseCanvasNode`, `BaseNodeSet`, `PortDef`, `NodeUIConfig`, `ConfigField`, `DisplayField`, or wire types affects every nodeset author, so it goes through an [ADR](docs/pages/developer-guide/core/decisions/index.html).

---

## 3. Dev loop & checks

For the full install flow (conda envs, submodules, data), read [`INSTALL.md`](docs-md/INSTALL.md). Once installed, the minimal dev loop is:

```bash
conda activate agentcanvas            # Python 3.10+, ADR-platform-004
cd agentcanvas && bash run_dev.sh     # backend :8000 + frontend :5173
bash docs/run_dev.sh                  # doc-site on :8092 (live reload)
```

Install pre-commit once after clone (`pip install pre-commit && pre-commit install`), then run the full check set before you push:

- `pre-commit run --all-files` — whitespace, EOF newline, YAML/JSON validity, file-size limit
- **Backend + import boundary**: `cd agentcanvas/backend && python -m pytest`
- **Backend smoke**: `cd agentcanvas && bash run_dev.sh` — server starts and `/docs` loads
- **Frontend**: `cd agentcanvas/frontend && npm run build` (or `npm run typecheck`)
- **Docs**: no build step — open the page in `bash docs/run_dev.sh` (:8092) and confirm it renders

---

## 4. PR format

Branch from `master` (`feat|fix|refactor|docs/<short-desc>`), and keep **one concern per PR**. Commits follow Conventional Commits: `type(scope): description`, lowercase, imperative, ≤72 chars (`feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `ci`, `perf`). The style rules live in [`.claude/standard/`](.claude/standard/) — read the one for your task before writing code.

```markdown
## Summary
- What changed and why (1–3 bullets)

## Reference
Closes #N, or links the Discussion this was agreed in

## Test plan
- [ ] How you verified it works (pytest / npm run build output)
```

---

## Getting help

- **Roadmap**: [Roadmap](docs/pages/developer-guide/core/roadmap.html)
- **Talk before a big change**: [GitHub Discussions](https://github.com/jianzhou0420/AgentCanvas/discussions)
- **Who built what (+ citations)**: [Credits](docs/pages/developer-guide/community/credits.html) (doc-site) · README
- **What nodesets already exist**: [Nodeset docs](docs/pages/developer-guide/nodesets/index.html)
- **Glossary**: [glossary](docs/pages/developer-guide/core/glossary.html) — read before touching canvas / graph / state features
- **Design rationale**: [Architecture Decisions](docs/pages/developer-guide/core/decisions/index.html) — ADRs grouped by field
- **Repository layout**: [Repo guide](docs/pages/developer-guide/repo/index.html)
- **Issues**: [github.com/jianzhou0420/AgentCanvas/issues](https://github.com/jianzhou0420/AgentCanvas/issues)

# L3 agent 臂 v2 — 改动记录

针对 GPT 对 L3 loop 的设计审查（8666 `design/l3-loop-review.html`），逐条落实。
每条都注明**为什么改**（哪个测量逼出来的）和**改在哪**。

新臂 `--mode agent2`（arm `l3_agent2`），旧臂 `--mode agent` 原样保留，所以
n=100 的基线仍然可复现。

---

## 0. 先记一条审查本身的错误

GPT 全文的对照基线是 `L0 = SR 44 / OSR 49 / SPL 0.420 / 4783 步`。这四个数逐位
吻合 **`std_vla_l0_acdefault`** —— 那是 **rig 偏移实验**的 L0（512² / FOV 90 /
相机 1.25 m），不是 L3d 实际运行的训练 rig。

```
GPT 用的   std_vla_l0_acdefault   SR 44.0  OSR 49.0  SPL 0.420    512²/90°/1.25m
应该用的   vla_alone              SR 57.0  OSR 64.0  SPL 0.542    720×640/110°/0.5m
```

这个 rig 差我们自己测过：**−13 SR 点，p=0.029**。用正确基线后它的三条核心判断
全部反转：

| GPT 的结论 | 用正确基线 |
|---|---|
| L3 净赢 4 条 | **净输 9 条**（赢 15 / 丢 24，p=0.20） |
| Oracle SR **+8**，planner 能把机器人带到更多正确区域 | Oracle SR **−7**（57.0 vs 64.0） |
| "潜力已被数据证明了一半" | 这半个证明不存在 |

**它的工程批评仍然大部分成立**，下面逐条落实；但它的研究结论不成立。

### 一条保留的反对意见

GPT 建议里工程量最大的是 typed state + receipt 召回机制。我判断 ROI 不成立：

- L3d **OSR 57**。把终止/状态/验证全部做到完美，SR 上限 = OSR = **57.0**，
  正好等于 L0 现在的 SR。这些改动的天花板**等于基线**。
- 43 集失败里 **36 集从未进入过 3 m**。那是策略能力问题，任何 harness 都碰不到。

用户要求全修，已全部实现。这条判断留在这里，等 v2 的数字来裁决。

---

## 1. 验证器离线标定（P0-3 第一步）

先做的事，零成本，决定了后面所有改法。对着已有 100 集日志：

```
                真成功   真失败
  接受 ( 51)      31       20   ← 误收
  否决 ( 49)      17       32   ← 误拒
  一致率 63%   精确率 61%   召回 65%

误收 20 集: NE 中位 6.96 m,  7 集 >8 m（认错地方）, 6 集 3–5 m（差一点）
误拒 17 集: NE 全部 <3.0 m,  中位 1.80 m,  最小 0.54 m
            17/17 都做过 360° 扫视,  0/17 做过 verify_step
```

**误拒那 17 集是全 run 最锋利的信号**：机器人站在离目标中位 1.8 米处、已经看完
整圈全景，验证器仍然说"没到"。原因不是它瞎，是**问错了问题**。

---

## 2. P0 · 验证目标与 benchmark 定义不一致

**症状**：R2R-CE 的 success 是「离目标 < 3 m」，验证器问的是「你能不能看见并
点名那个地标」。两者不等价，双向错误由此而来。

**改法**（`agent_judge2.VERIFY_SYSTEM` + `adjudicate()`）：把单个 `satisfied`
拆成它一直在偷偷平均的三个问题，**并把合成规则搬回代码里**：

```
target_visible   看得见并说得出在哪一帧吗
relation_holds   空间关系成立吗（"在吧台拐角"≠ 隔着房间看着它）
close_enough     离那个地方几米之内吗   ← 这一条决定分数
confidence       0..1
```

`adjudicate()` 的规则直接对着混淆矩阵标定，而不是凭直觉：

- 误拒那 17 集**全部近距离 + 已扫视 + 认不出** → `close_enough` 在扫视后**单独
  即可判定到达**。要求点名地标正是把这些扔掉的原因。
- 误收那 20 集**看得见但远**（中位 6.96 m）→ `target_visible` 单独**永不充分**，
  必须同时断言距离。
- **点名的可见矛盾一律推翻一切** —— 这条是用一集结束在浴室走廊换来的。

规则在代码里而不在提示词里，所以以后可以**离线拿日志重新标定，不用重跑**。

---

## 3. P0 · 终止三态，强制停止不再等于验证通过

**症状**：否决额度用尽后 `finish(forced=True)`，与真正的验证通过记在同一个字段
里 —— 100 集里 48 集是这样结束的，统计上无法区分。

**改法**（`agent_judge2` + `agent_loop2`）：

```
VERIFIED_SUCCESS   验证器的观察合成为「到了」
VERIFIED_FAILURE   验证器点名了矛盾，或三项都不支持
UNRESOLVED_STOP    否决额度用尽 / 轮数耗尽 / 验证根本没返回
```

`UNRESOLVED_STOP` 仍然会发 STOP（不发就是 0 分），但**不计入验证器的正样本**。
`outcome` / `outcome_why` 进 result row。

## 4. P0 · fail-open 修复

**症状**：`parse_failed → {"satisfied": True}`。最不可逆的那道门，默认放行。

**改法**（`agent_judge2.ask()`）：严格 schema 校验（`require=` 指定必需字段）
→ 同模型重试一次 → **换 `claude-haiku-4-5` 备用验证器**再试 → 仍失败返回
`None`，由 `adjudicate()` 映射到 `UNRESOLVED_STOP`。

调用方**必须显式分支**：`None` 是唯一不能当成值用的返回。

> 补一句实测：上一轮 805 次调用 **0 次失败**，这条路径从未触发。它是潜在风险，
> 不是 n=100 那个结果的成因。

## 5. P0 · 开环校正改成微闭环

**症状**：验证器一次最多 12 个盲动作，执行完才重新观察。实测：

```
做过 verify_step 的 17 集:  SR 41.2%   L0 同集 64.7%   Δ −23.5
没做过的          83 集:  SR 49.4%   L0 同集 55.4%   Δ  −6.0
```

**改法**：

- 一次最多 `MICRO_ACTIONS = 2` 个动作，**之后立刻重新观察并重新提问**。
- **前进默认关闭**（`--ablate forward` 之外还可用 `--ablate verify-step` 整条关掉）。
  转身不改坐标，前进会把机器人从它可能已经身处的成功半径里走出去。
- 单独计数 `verify_turn_steps` / `verify_forward_steps`，并在每次校正**前后**
  用 `probe_distance()` 记录真实目标距离，产出 `delta_m`。

> `probe_distance()` 走 `env_habitat__evaluate`，读 habitat 的测量缓存、**不推进
> 模拟器**。这是 oracle 信息，**只进日志，绝不进模型上下文，控制流不得据此分支**。
> 它的存在是为了让"这次校正到底把机器人推近了还是推远了"有答案而不是猜测。

## 6. P0 · 状态写入的证据门（typed state）

**症状**：`rewrite()` 只检查是不是 dict、空不空、超不超长。任何模型生成的
`ruled_out` 都能直接成为长期状态，没有证据、来源、时间、置信度。而且空串不能
清除旧字段，被推翻的信念难以显式删除。

**改法**（`agent_state2.EvidenceState`）：五个自由文本字段 → 类型化信念记录。

```
Belief: id · claim · kind(observed|inferred|negative) · evidence_ids
        created_turn · last_confirmed_turn · confidence · status(active|contradicted|retired)
```

**负事实的不对称门**是这块的核心：

> 正事实错了，走过去就能纠正；**"那边不通"一旦写错，agent 可能永远不再探索。**

所以 `kind == "negative"` 的信念**必须**引用 harness 真实记录过的帧 id，或者
机器人刚做过 360° 扫视；否则**拒绝写入、记进 `rejected_writes` 并渲染给模型看**。

另外：`confirm` / `contradict` / `retire` **按 id 操作**。模型不再需要正确复述
一整段散文才能保住一个事实，也不会因为忘记复述而悄悄丢掉一个。

`--ablate evidence` 可以关掉这道门，做 A/B。

## 7. P1 · receipt：证据保全的记忆

**症状**：只给模型看最近 6 条 ledger，更早的在磁盘上但运行中召不回。代码注释里
那句「更早的 dispatch 如果重要，应该已经在 PROGRESS 里」不成立 —— PROGRESS 是
模型写的可错摘要，不是 harness 事实。

**改法**：review 返回 receipt：

```
receipt: { claim, evidence_ids, incidental, not_done }
```

每一帧在进模型时都带 `id=`（事件日志写盘的文件句柄），所以一条主张事后可以
解析回真实像素。上下文仍然有界。

## 8. P1 · 失速检测不再只看净位移

**症状**：判据是 `net_displacement_m < 0.3`，它把撞墙、立刻 STOP、**合理的原地
转向**、绕回原处、pose 丢失全混在一起。

**改法**：habitat 的 `step_discrete` 不暴露碰撞，所以用**帧相似度**替代 ——
`VlaToolSet.dhash()` 算 64 位差分哈希（本地、免费、不用模型）：

```python
wedged = consecutive_stalls >= 3 AND consecutive_same_view >= 2
```

**「转身看到了新东西」不再判 wedged；「不动且画面没变」才是。**
状态块里的警告文案也据此分叉。

（测试里踩到一个自己的坑：`1` 和 `2**40` 只差 2 个比特，在汉明距离下它们**是**
同一个视图 —— 数值相距多远与视觉相似度无关。）

## 9. P1 · 缺失 next_instruction 不再静默重放整条 mission

**症状**（**这条是我自己在核对 GPT 的 review 时查出来的，它没提到**）：

```
完整版:    子指令 == 原始 mission 的派发    0 / 306 次
no-drive:  子指令 == 原始 mission 的派发  371 / 643 次 · 29 集 · 单集最多 23 次
```

拆开看，之前"drive 是止损"那个结论站不住：

| | 集数 | no-drive | 完整版 |
|---|---|---|---|
| 触发了 fallback 重放 | 29 | 27.6% | 44.8% |
| **没触发** | **71** | **47.9%** | **49.3%** |

**在 71 集干净的集上，有没有马达几乎没差别。** 那 −6.0 全部来自被污染的 29 集
—— no-drive 那次消融测的是"没有马达 **+ 一个退化的重放循环**"。**该消融必须重跑。**

**改法**：显式状态流转，没有静默回退：

```
missing_next_instruction
    ├─ near_goal        → 进入验证
    ├─ review 解析失败   → 重试 review（最多 2 次）
    ├─ 还有 replan 额度  → 用 next_objective 做一次受限重规划
    └─ 否则             → 结束派发（记 ended_dispatching）
```

## 10. P1 · 到达闸收紧

**症状**：只要 policy_stop + 位移 ≥2 m + 有一句观察 + 没写矛盾，就能把 continue
改成 finish。中间子目标（"走到厨房"）成功停下也会被过早送进验证。

**改法**：

```
policy_stop
AND 位移 ≥ 2 m
AND 有观察（receipt.claim 非空）
AND near_goal
AND terminal_clause_active      ← 新增，靠 clause 进度表判定
AND 没有点名矛盾
```

`terminal_clause_active` 由 bootstrap 拆出的 clause 列表 + review 标记的进度
推导。这也是为什么 v2 让 bootstrap 输出 `clauses[]`。

## 11. P2 · 预算与日志语义统一

- **STOP token 的步数语义**：`steps_used` 曾经含 STOP、`steps_taken_total` 不含，
  同一个 telemetry 里"步"有两个意思。现在 `steps_used` 只数环境步，另出
  `generations` 记模型生成次数。
- **四套独立计数**（`toolset`）：
  `policy_env_steps` / `harness_drive_steps` / `verification_steps` /
  `render_only_observations`（全景是渲染，不花环境步）。
- **调用账本**（`trace["counts"]`）：`planner_calls` / `review_calls` /
  `verify_calls` / `parse_failures` / `forced_stops` / `unverified_stops` /
  `verify_looks` / `verify_steps` / `verify_turn_steps` / `verify_forward_steps` /
  `state_writes_accepted` / `state_writes_refused`。
- **`FINISH_MARGIN` 不再写死**：由 `max_steps * 1.2` 推导，跟着 `--max-steps` 走。
- **上下文长度**：`ctx_tok = input + cache_read + cache_creation`。上一版只记了
  前两项，**图片整个落在漏掉的第三项里**，导致统计出来"14 张图占 0 token"。

## 12. P2 · 仓库内的状态机测试

`vlaharness/tests/test_agent_loop2.py` —— 假 toolset + 脚本化模型回复，**18 个
测试全通过**，锁住：

- 三种终止状态各自的路径
- 验证失败不放行 / review 失败不开到达闸
- 到达闸要求 near_goal + terminal clause
- **不重放 mission**、马达指令不进文本通道
- 转身看到新东西不算 wedged / 画面冻结才算
- 校正 ≤2 动作、`forward` 与 `drive` 消融确实生效
- 负事实无证据被拒 / 有引用被接受 / 按 id retire
- 渲染长度有界
- `adjudicate()` 对四种标定形态的判定

> 这套测试的存在理由：这个项目最贵的东西是 100 集的跑分，而**迄今每一个
> 让我们付出一次跑分的 bug，都是一个假机器人在一秒内能触发的控制流 bug**。

---

## 消融轴（v2）

```
verify        不做验证调用
sweep         验证器不能用免费 360° 扫视
drive         完全禁用马达
verify-step   验证器可以看但不能动     ← 新增，单独隔离校正通道
forward       验证器校正只能转不能走   ← 新增
back          三视图（退回 270° 盲区）
frames        每段 3 帧而不是 10 帧
evidence      关掉证据门（旧行为）     ← 新增，A/B 用
```

## 文件

| 文件 | 状态 |
|---|---|
| `agent_state2.py` | 新增 —— `EvidenceState` |
| `agent_judge2.py` | 新增 —— 三个调用 + `ask()` + `adjudicate()` |
| `agent_loop2.py` | 新增 —— v2 循环 |
| `tests/test_agent_loop2.py` | 新增 —— 18 个状态机测试 |
| `toolset.py` | 改 —— 四套计数 · `probe_distance()` · `dhash()` · STOP 语义 |
| `run.py` | 改 —— `--mode agent2` · 新消融轴 · outcome/counts 进 row |
| `agent_state.py` / `agent_judge.py` / `agent_loop.py` | **未动**，v1 仍可复现 |

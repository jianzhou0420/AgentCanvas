# M1 蒸馏候选池（eharness-evo）

从 R2R rand100 **bare 老师板**（`std_sdk_opus-5_bare_default`，SR 0.69）的成功集蒸馏出的
briefing/skill 文本候选，目标学生 = Qwen3.5-9B bare（已知地板 SR≈5–7%）。

## 合同

- 一个候选 = 一个 `cand_NN_<slug>.md` 文件；`skill_text` 段落即将来 fork 学生臂时
  **逐字追加进 briefing 的那段文本**（英文、原子、场景无关——无坐标/无场景名/无集特定物体）。
- 上臂纪律：一次 fork 只带**恰好一条** skill_text（原子 delta，可归因）。
- 判卷：`labs/evolution/paired_gate.py`（同集配对、discordant-only 单侧精确 McNemar，
  α=0.025）。绝对"救回 N 集"判据禁用。
- 首批判决设计：3–5 条候选若 0 条过门 → 按 transcript 判词分类
  （没读 / 读了不执行 / 执行了没用）——这本身是"9B 能不能被文本提起来"的第一个答案。

## 已知先验

- ICL 实验（07-19）：fable-episode 整集示范帮 qwen-4b（batching 行为迁移成功）、
  **打残 qwen-27b**（提前停）。→ 候选文本宜短、宜原子；尺寸是风险旋钮。
- wp 实验（07-15）：4b 拿着 anti-circling skill **不执行**。→ 判词分类里
  "读了不执行"是真实存在的失败模式，候选必须机械可执行（触发→动作），不能依赖判断力。

## 候选池与首批建议（08-22 挖掘完成：8 成功集 + 2 失败集 + 2 学生冒烟逐字读）

| # | 候选 | 频次/依据 | 首批 |
|---|------|-----------|------|
| 01 | batching 动作成批 | 用户观察 + ICL 迁移先验 + 老师 6.43/次 | ✅ |
| 02 | scan-before-walk 先扫再走 | 6/8 成功集，最机械 | ✅ |
| 04 | final-approach-stop 终推果断停 | 5/8 + **直击学生两大终局败形**（run2 穿过终点不敢停 oracle=1.0 判负；run1 验证循环烧 495 步） | ✅ |
| 03 | center-then-advance 居中推进 | 5/8 | 二批 |
| 05 | turn-calibration 转向查表 | 5/8；**上臂前置检查真转角=15°**（RxR 口径混淆教训） | 二批 |
| 06 | blocked-then-pivot 撞墙即绕 | 4/8，触发二元 | 二批 |
| 07 | clause-ledger 子句清单 | 6/8 但幻觉进度风险最高（run1 开局即宣称"已完成若干段"） | 观察 |

老师失败集对照：ep40（495 步走偏 23.5m）正是 02/07 防的病；ep2（被 3.7m 外水槽骗停）是 04 的残余风险。

## 材料口径备注

- 老师板每次 step 平均 6.43 动作（成功集内 6.07）；本地 9B 冒烟 4.6–4.7（n=3 集，样本极小）。
  **口径复核（08-22）**：本地 9B 板确为裸 bare（bare:true、observe+step、15° 转向）——
  与老师 bare 同构可比；gen-0 学生臂配置即此 cell（`std_eharness_qwen3.5-9b_bare`），无需新造。
  用户在 h100 百集板上观察到接近 1 动作/次的极端行为——待拿到该板档案后对账。
- RxR bare 老师板存在"真转 30°/prompt 写 15°"口径混淆（outputs-deepdive 实锤），
  蒸馏材料**只用 R2R** bare 板。

# cand_01 · batching（动作成批规划）

- **pattern**: 老师一次 `step` 调用规划一串动作（如 `[2,2,1,1,1,1]`：先转向对准，
  再连走数步），每次调用平均 6.4 个动作；观察-决策的节奏是"看一眼 → 走一段"，
  而不是"看一眼 → 挪一下"。学生（9B）在大板上被观察到接近一次一个动作——
  每步都重新看图重新想，既慢又容易在原地振荡。
- **skill_text**（追加进学生 briefing 的逐字文本）:

  > Plan your movement in batches, not single actions. In every step() call,
  > issue a SEQUENCE of actions: first the turns to face your chosen direction,
  > then several forward moves along it — for example step([2,2,1,1,1,1]).
  > A typical batch is 4-8 actions. Issue a single action only when you are
  > about to STOP or squeezing through a tight doorway. After each batch,
  > look at the new view and plan the next batch.

- **evidence**:
  - 定量：老师板 2609 次 step 均 6.43 动作/次（成功集内 6.07），1 动作调用仅 4%；
    本地 9B 冒烟 4.6/次、1 动作 8%（n=3 集，样本小）。**口径已复核（08-22）**：本地
    9B 板确为裸 bare（config bare:true、工具仅 observe+step、prompt 无 veer/depth），
    与老师 bare 可比——此前"eharness 臂不可比"的注记是挖掘 agent 误读 cell 名，已纠正。
    用户报告 h100 百集板接近 1 动作/次，上臂后以学生自身百集档案回填真实分布。
  - 先验：ICL 实验（07-19）中 batching 正是通过示范成功迁移给 qwen-4b 的行为——
    这条技能有"小模型学得会"的直接证据。
- **why_student_executable**: 纯机械规则（每次调用给 4-8 个动作），不需要任何场景判断。
- **risk**: 批太长会放过被挡/走偏（盲走 8 步才看图）；措辞已含"贴近 STOP/窄门时单步"缓冲。
  若判词显示大量批中撞墙，下一轮候选改批长上限而非撤销方向。
- **status**: 待上臂（fork 学生 bare 臂 + 此段文本 → 筛选板 → 确认板）

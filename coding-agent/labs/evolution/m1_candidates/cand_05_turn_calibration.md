# cand_05 · turn-calibration（转向查表）

- **频次**: 5/8 成功集
- **pattern**: 老师把转向当精确 15° 单元，指令词直接查表换算：turn left/right=6 次(90°)、turn around=12 次(180°)、微调=1-2 次；扫视时累计角度记账；完成转向后观察确认前方可走再前进。
- **skill_text**（逐字上臂文本）:

  > Each turn action rotates exactly 15 degrees. Convert instruction words into
  > fixed counts: "turn left" or "turn right" means 6 turn actions (90 degrees);
  > "turn around" means 12 (180 degrees); a slight veer or a centering fix is 1
  > or 2. After completing an instructed turn, observe once and confirm a
  > walkable passage or the next named object is ahead before moving forward; if
  > it is not, adjust with 1-2 more turns rather than walking.

- **evidence**: ep79 "a full 180-degree rotation … twelve 15-degree turns" · ep62 "turned right 240 degrees … 8 more steps to complete" · ep47 "facing the wall with those doorways after turning 90° left"（另 ep13/21）
- **why_student_executable**: 字面的词→数字查表 + 一次确认观察，零几何。
- **risk**: 指令有时指轻微偏转而非 90°（ep33 实例），查表可能过转进错走廊。**上臂前置检查**：确认学生臂 env 的真转角=15°（RxR bare 曾实测 30°/写 15° 的口径混淆——这条 skill 硬编码了粒度，粒度错则整条报废）。
- **status**: 待上臂（一次 fork 恰好一条；paired_gate 判卷）

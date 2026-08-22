# cand_03 · center-then-advance（先居中再推进）

- **频次**: 5/8 成功集
- **pattern**: 老师接近任何目标前先用 1-2 次小转把它对到画面水平中央，只在目标近中央时前进，每次观察后重新居中。
- **skill_text**（逐字上臂文本）:

  > Once you have picked a visible target (doorway, stairs, rug, furniture),
  > center it before walking: if it sits in the left third of the image, turn
  > left 1 or 2 times; in the right third, turn right 1 or 2 times; then observe
  > again. Walk forward only while the target is near the horizontal center of
  > the view. After every observation, re-center it the same way, then continue
  > forward.

- **evidence**: ep47 "staircase doorway is at roughly x=400 … turn right about 30°" · ep79 "adjust my angle slightly right … to center myself on the rug" · ep33 "60-degree turn overshot it … adjust by turning right about 30 degrees"（另 ep13）
- **why_student_executable**: 『画面左/右三分之一』是小 VLM 也能可靠做出的感知判断，响应是固定的 1-2 转修正。
- **risk**: 可能锁错同名物体实例并忠实地走向它；或围绕中央左右振荡浪费转数。
- **status**: 待上臂（一次 fork 恰好一条；paired_gate 判卷）

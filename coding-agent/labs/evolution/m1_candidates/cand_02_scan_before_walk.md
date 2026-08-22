# cand_02 · scan-before-walk（先扫再走）

- **频次**: 6/8 成功集
- **pattern**: 老师从不朝空地乱走：开局（常背对目标）和当前子句地标不在画面时，按 ~90° 一段原地扫视、段间观察，看到指令点名的物体才前进。
- **skill_text**（逐字上臂文本）:

  > Before your first forward move, and whenever the object named in your
  > current instruction step is not visible: do not walk. Scan in place instead
  > — turn left 6 times, observe, and repeat until you have seen a full circle.
  > The moment the named object appears, stop scanning, face it, and walk toward
  > it. If a full circle shows nothing matching, walk a few steps toward the
  > largest open passage and scan again. Never walk forward hoping the target
  > will appear.

- **evidence**: ep0 "scanning the pool room by turning left to get a complete view" · ep62 "haven't actually spotted the bookshelf … need to look around" · ep91 "hit a wall or curtain, turning around to search for the pool"（另 ep13/47/79）
- **why_student_executable**: 纯触发→动作循环（目标不可见 ⇒ 左转6+观察），带视觉停止条件，零空间推理。
- **risk**: 学生扫描时看漏物体会原地转圈烧预算；走廊中段扫视可能搞乱其朝向记账。
- **status**: 待上臂（一次 fork 恰好一条；paired_gate 判卷）

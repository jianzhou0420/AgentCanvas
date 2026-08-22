# cand_04 · final-approach-stop（终推与果断 STOP）

- **频次**: 5/8 成功集 + 直击学生两大终局败形
- **pattern**: 老师从不初见即停、也从不盲停：靠近到终点物体大且近（1-3m），再补 2-3 步（进房/到门槛/贴柜台），确认一眼，立即 STOP，不再探索。学生 run2 穿过终点从未敢停（oracle_success=1.0 判负）、run1 在验证循环里烧 495 步——这条对症。
- **skill_text**（逐字上臂文本）:

  > Do not issue STOP while the final object or room named in the instruction is
  > out of view. When you can see it, keep walking toward it until it is close
  > and fills a large part of the image, then take 2 or 3 more forward steps —
  > into the room if told to enter, right up to the object if told to stop near
  > it. Observe once to confirm it is directly ahead or beside you, then issue
  > STOP immediately. Do not keep exploring after this confirmation.

- **evidence**: ep13 "inside the bedroom now and need to move a bit further in before stopping" · ep62 "about 2-3 meters from the white couch … advancing slightly would be better" · ep91 "reached the bar counter … within the 3-meter range"（另 ep0/21）
- **why_student_executable**: 『物体大且居中 → 补2-3步 → 确认 → STOP』是固定终局序列，替换学生开放式的『我到了吗』循环。
- **risk**: 『占画面很大』可能被大号的错误实例提前触发（老师失败集 ep2 即被 3.7m 外的水槽骗停）——把差点成功变成自信的错停。
- **status**: 待上臂（一次 fork 恰好一条；paired_gate 判卷）

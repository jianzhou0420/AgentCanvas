# cand_06 · blocked-then-pivot（撞墙即绕）

- **频次**: 4/8 成功集
- **pattern**: 老师用视觉检测被挡（发了前进但画面没变——step 结果仍报全部 executed，图像是唯一信号），立即向更开阔一侧转 30-90° 绕行，绝不重发同样的前进串。
- **skill_text**（逐字上臂文本）:

  > After every step call, compare the new view to the previous one. If you sent
  > forward actions but the view did not change, you are blocked by a wall or
  > furniture. Never resend the same forward sequence. Instead, turn 2 or 3
  > actions toward whichever side of the image shows more open floor, move
  > forward 2 to 4 steps, observe, and then re-center your original target. If
  > you are still blocked, try the opposite side the same way.

- **evidence**: ep33 "movement command didn't register—hitting a collision boundary" → "experiment with moving slightly to the left instead" · ep21 "pressed against the left wall, pivot right"（另 ep0/62）
- **why_student_executable**: 『前进后画面没变』是二元感知检查，恢复动作是固定的转-走-看宏。
- **risk**: 9B 在低纹理墙上可能误判『没变』（假阳性→之字走）；或真被挡却没察觉，又被禁止重发前进，导致过早放弃正确方向。
- **status**: 待上臂（一次 fork 恰好一条；paired_gate 判卷）

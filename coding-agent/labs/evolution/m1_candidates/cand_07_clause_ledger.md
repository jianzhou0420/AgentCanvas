# cand_07 · clause-ledger（子句清单）

- **频次**: 6/8 成功集但风险最高
- **pattern**: 老师几乎每次观察后都复述指令进度（刚完成哪句、当前哪句），一次只执行一句。
- **skill_text**（逐字上臂文本）:

  > Treat the instruction as a numbered checklist of steps, in order. In every
  > reply, write the checklist and mark each step DONE or CURRENT. Work only on
  > the CURRENT step; mark it DONE only after its named object or turn has
  > actually appeared in your camera view and been reached — never because you
  > assume it happened. When the last step is marked DONE, follow your stopping
  > procedure. If you become lost, return to the last step you are certain was
  > DONE.

- **evidence**: ep13 "I'm at the front door now and need to turn left to find the stairs" · ep21 "This bedroom … matches the instruction to enter the third entryway on the left"（另 ep47/62/79）
- **why_student_executable**: 写清单是死记硬背的文本习惯；DONE 门（物体必须真出现在画面里）把推进变成感知检查而非判断。
- **risk**: 弱模型幻觉进度——学生 run1 第一条 thinking 就宣称『我已完成若干段』；清单可能给跳句盖章合法化，且每轮多耗 token。排名垫底即因此。
- **status**: 待上臂（一次 fork 恰好一条；paired_gate 判卷）

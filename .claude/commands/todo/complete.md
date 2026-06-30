# Complete a TODO item

Mark a TODO item as completed in `docs/core/roadmap.md`.

## Input

The user provides a TODO ID as `$ARGUMENTS` (e.g. `3`). If no arguments are provided, run `/fetchtodo` first and ask which item to complete.

## Steps

1. Read `docs/core/roadmap.md`
2. Find the `## TODO` section
3. Find the line matching `- [ ] **$ARGUMENTS**` (the bold ID number)
4. Change `- [ ]` to `- [x]` on that line
5. Update the `last updated:` timestamp at the top using `date '+%Y-%m-%d %H:%M'`
6. Print confirmation: `Completed #$ARGUMENTS: <item text>`
7. If no match found, print the TODO list and say "No TODO with ID $ARGUMENTS"

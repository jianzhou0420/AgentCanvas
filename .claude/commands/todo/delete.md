# Delete a TODO item

Remove a TODO item from `docs/core/roadmap.md`.

## Input

The user provides a TODO ID as `$ARGUMENTS` (e.g. `3`). If no arguments are provided, run `/fetchtodo` first and ask which item to delete.

## Steps

1. Read `docs/core/roadmap.md`
2. Find the `## TODO` section
3. Find the line matching `- [ ] **$ARGUMENTS**` or `- [x] **$ARGUMENTS**` (the bold ID number)
4. Remove that line entirely
5. Do NOT renumber other items — IDs are stable and never reused
6. Update the `last updated:` timestamp at the top using `date '+%Y-%m-%d %H:%M'`
7. Print confirmation: `Deleted #$ARGUMENTS: <item text>`
8. If no match found, print the TODO list and say "No TODO with ID $ARGUMENTS"

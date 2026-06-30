# Add a TODO item

Add a new TODO item to the TODO section in `docs/core/roadmap.md`.

## Input

The user provides a description of the TODO item as `$ARGUMENTS`. If no arguments are provided, ask the user what to add.

## Steps

1. Read `docs/core/roadmap.md`
2. Find the `## TODO` section
3. Read the `<!-- next_id: N -->` comment to get the next available ID
4. Add `- [ ] **N** $ARGUMENTS` as a new line at the end of the TODO checklist (before any blank line or next section)
5. Update the `<!-- next_id: N -->` comment to `<!-- next_id: N+1 -->`
6. Update the `last updated:` timestamp at the top using `date '+%Y-%m-%d %H:%M'`
7. Print confirmation: `Added TODO #N: $ARGUMENTS`

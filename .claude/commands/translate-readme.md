# Translate README

Sync all translated README files with the current English `README.md`.

## Target files

| File | Language |
|------|----------|
| `README_zh.md` | Chinese (Simplified) |
| `README_es.md` | Spanish |
| `README_ja.md` | Japanese |
| `README_ko.md` | Korean |

## Steps

1. Read `README.md` to get the current English source.
2. For each target language, **launch a parallel executor agent** that:
   - Reads the existing translated file (if it exists) for reference
   - Writes the updated translation to the target file
3. After all agents complete, verify all 5 files have a language switcher on line 1 and matching structure.

## Translation rules (pass these to each agent)

1. Translate ALL prose text. Keep the same structure, tone, and emphasis.
2. **DO NOT translate**: code blocks (Python, JSON, bash), file paths, class names, technical identifiers (`BaseCanvasNode`, `PortDef`, `NodeSet`, etc.), URLs, command examples.
3. Keep the same markdown formatting (headers, tables, code fences, blockquotes).
4. Each file must have a language switcher on line 1:
   - English: `**English** | [中文](README_zh.md) | [Espanol](README_es.md) | [日本語](README_ja.md) | [한국어](README_ko.md)`
   - Chinese: `[English](README.md) | **中文** | [Espanol](README_es.md) | [日本語](README_ja.md) | [한국어](README_ko.md)`
   - Spanish: `[English](README.md) | [中文](README_zh.md) | **Espanol** | [日本語](README_ja.md) | [한국어](README_ko.md)`
   - Japanese: `[English](README.md) | [中文](README_zh.md) | [Espanol](README_es.md) | **日本語** | [한국어](README_ko.md)`
   - Korean: `[English](README.md) | [中文](README_zh.md) | [Espanol](README_es.md) | [日本語](README_ja.md) | **한국어**`
5. Technical terms use **first-mention pattern**: native translation followed by English in parentheses. e.g., "响应式执行引擎 (Graph Execution Engine)".
6. Taglines should be compelling in the target language, not word-for-word.
7. Keep code comments in English.
8. Do NOT add or remove any content relative to the English source.

## After completion

Print a summary table:
```
File            Lines  Status
README.md       269   source
README_zh.md    269   updated
README_es.md    269   updated
README_ja.md    269   updated
README_ko.md    269   updated
```

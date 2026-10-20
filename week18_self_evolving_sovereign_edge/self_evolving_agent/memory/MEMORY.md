# MEMORY.md — World & Project Facts
# Maintained by the agent's background consolidator (Part 6).

## Project Context
(none learned yet)

## Known Issues
(none learned yet)

## Technical Decisions
- Deferred tools (Read, Edit, Write, Glob, etc.) must be fetched via `ToolSearch { query: "select:ToolName" }` before use — their parameter schemas are unknown otherwise, leading to InputValidationError.
- `Read`/`Edit`/`Write` all use `file_path` (absolute path) as the primary file parameter.
- `Edit` requires `old_string` + `new_string`; calling it with `new_text` or similar causes failure.

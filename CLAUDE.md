# Trueform Claude Router

@AGENTS.md is the authority for every agent working in this repository. This
file exists only to route Claude-based agents into that contract; it is not a
second style guide, execution guide, or command reference.

## Required route

1. Read @AGENTS.md.
2. Follow its **Read first** order exactly.
3. Read only the task-specific references named there.

## Task references

- C++ implementation: @agents/cpp_core_architecture.md and
  @agents/cpp_engineering_philosophy.md.
- CSG correctness: @agents/csg_pipeline_debugging.md.
- Python/nanobind: @agents/python_layer.md.
- TypeScript/WASM: @agents/typescript_layer.md.
- Public cross-language feature: @agents/feature_lifecycle.md.
- Documentation: @agents/documentation_architecture.md.
- Caller-facing usage: the relevant `agents/usage_*.md` file and
  @agents/cpp_modules.md for C++ API lookup.

The role files under `.claude/agents/` are also routers. @AGENTS.md and the
task-specific references remain authoritative.

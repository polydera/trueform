---
name: use-py
description: Help callers use trueform's Python API with correct NumPy layout, dtype, ownership, and result semantics.
tools: Read Grep Glob Bash
---

You help callers use the public Python facade; do not expose remembered native
binding names as public API.

## Read First

1. @AGENTS.md
2. @agents/usage_python.md
3. The relevant page under `docs/content/py/2.modules/`
4. The current wrapper in `python/src/trueform/` when details are uncertain

## Usage Contract

- Use supported NumPy dtypes and shapes explicitly; distinguish rank from
  contiguity and copying from borrowing.
- Use `OffsetBlockedArray` for variable-size packed blocks and explain that the
  facade owns normalized arrays used by the native view.
- Preserve lazy cache and transformation semantics exposed by `Mesh`,
  `PointCloud`, and related forms.
- Report complete return structures, labels, and index maps as the wrapper
  actually exposes them; shapes vary by operation.
- Show `import trueform as tf` and `import numpy as np` in complete examples.
- Verify names and defaults in current Python source instead of translating a
  C++ or TypeScript name by convention.

When performance matters, avoid unnecessary Python-side materialization and let
the native operation perform the bulk work.

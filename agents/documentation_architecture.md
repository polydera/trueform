# Documentation Architecture

Trueform's documentation is built with Nuxt Content, serving three parallel language tracks (C++, TypeScript, Python) from a single docs site at `trueform.polydera.com`.

---

## 1. Directory Structure

```
docs/
├── content/
│   ├── index.md                    # Landing page
│   ├── cpp/
│   │   ├── 1.getting-started/      # Installation, intro, live examples
│   │   ├── 2.modules/              # Per-module API reference
│   │   ├── 3.benchmarks/           # Per-module benchmark pages
│   │   ├── 4.vtk/                  # VTK integration guide
│   │   ├── 5.examples/             # Guided walkthroughs
│   │   └── 6.about/                # Research, contributing, license
│   ├── ts/
│   │   ├── 1.getting-started/      # Same structure
│   │   ├── 2.modules/              # Same 10 modules
│   │   ├── 3.benchmarks/
│   │   ├── 4.examples/
│   │   └── 5.about/
│   └── py/
│       ├── 1.getting-started/      # Same structure
│       ├── 2.modules/              # Same 10 modules
│       ├── 3.benchmarks/
│       ├── 4.blender/              # Blender-specific
│       ├── 5.examples/
│       └── 6.about/
├── app/                            # Nuxt app components
├── modules/                        # Custom Nuxt modules
├── public/                         # Static assets
├── wasm-examples/                  # Interactive WASM demos
├── nuxt.config.ts
└── content.config.ts
```

---

## 2. Module Documentation Pattern

Each language has a module doc per C++ module (`2.modules/01.core.md` through `<NN>.io.md`), covering the same modules but with language-appropriate API, examples, and idioms. Use `Glob` with `docs/content/cpp/2.modules/*.md` to see the current list.

### Frontmatter
```yaml
---
title: Core
description: Primitives, ranges, transformations, queries, buffers, policies, and algorithms.
navigation:
  icon: i-lucide-atom
---
```

### Section Structure (per module)
1. **Introduction** — one paragraph explaining the module's purpose
2. **Include/import** — how to use the module
3. **Types** — data structures, classes, with tables
4. **Functions** — grouped by category (e.g., "Mesh Generation", "Registration", "Edge Analysis")
5. **Examples** — inline code blocks showing real usage

### Function Documentation Pattern

**C++** (`docs/content/cpp/2.modules/`):
```markdown
### make_sharp_edges

```cpp
auto edges = tf::make_sharp_edges(polygons, tf::deg(30.f));
// edges: blocked_buffer<Index, 2> — each pair is [v0, v1]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `polygons` | `tf::polygons<Policy>` | Input mesh |
| `angle_threshold` | `tf::rad<T>` or `tf::deg<T>` | Dihedral angle threshold |
```

**TypeScript** (`docs/content/ts/2.modules/`):
```markdown
### Sharp Edges

```ts
const edges = tf.sharpEdges(mesh, 30);  // 30°, returns NDArrayInt32 [N, 2]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh` | `Mesh` | Input mesh |
| `angleDeg` | `number` | Angle threshold in degrees |
```

**Python** (`docs/content/py/2.modules/`):
```markdown
### sharp_edges

```python
edges = tf.sharp_edges(mesh, angle_deg=30)
# edges: np.ndarray shape (N, 2), dtype=int32
```
```

### Naming Across Languages

| C++ | TypeScript | Python |
|-----|-----------|--------|
| `make_sharp_edges` | `sharpEdges` | `sharp_edges` |
| `make_boolean` | `booleanUnion` | `boolean_union` |
| `neighbor_search` | `neighborSearch` | `neighbor_search` |
| `polygons_buffer` | `Mesh` | `Mesh` |

---

## 3. Custom Components

The docs use custom Vue/Nuxt components for rich content:

- `::tip{icon="..."}` — Info/warning callouts
- `::card-group` / `:::card` — Linked card grids for navigation
- `::try-it-banner` — Link to live interactive examples
- `::ts-query-diagram` — Visual diagram of the TS query system
- `::ts-flow-diagram` — Visual diagram of the TS data flow

---

## 4. Language-Specific Sections

**C++ only**: VTK integration guide (4.vtk/) — covers VTK adapters, filters, and examples.

**Python only**: Blender integration (4.blender/) — covers mesh conversion, scene management, and plugin development.

**TypeScript only**: Live examples are WASM-powered interactive demos embedded in the getting-started page.

---

## 5. Benchmark Pages

Each benchmark page (`3.benchmarks/`) documents:
- Hardware: Apple M4 Max, 16 threads
- Methodology: timing, datasets, comparison targets
- Results: tables and/or interactive charts
- Source code: links to benchmark source files

C++ has per-module benchmark pages (spatial, topology, geometry, remesh, intersect, cut, io). TS and Python have a single summary page.

---

## 6. Adding Documentation for a New Feature

1. **C++ docs**: Add to `docs/content/cpp/2.modules/<NN>.<module>.md`
   - Function name as `###` heading
   - Code block with example
   - Parameter table
   - Return type description

2. **TS docs**: Add to `docs/content/ts/2.modules/<NN>.<module>.md`
   - Same structure, camelCase naming
   - TS code block with types
   - Note async variant: `await tf.async.<function>(...)`

3. **Python docs**: Add to `docs/content/py/2.modules/<NN>.<module>.md`
   - Same structure, snake_case naming
   - Python code block with numpy types
   - Note OffsetBlockedArray for dynamic results

Ensure all three language docs are updated together when adding a feature.

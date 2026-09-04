## trueform v0.10.3

The vendored mimalloc is 3.5.1, built with large pages off. Upstream turned
them on by default in 3.1.6, pooling 64-512 KiB blocks — the size of most of
the engine's buffers — into shared 4 MiB pages that a partially used
per-thread heap cannot purge, so a long-lived multi-threaded process retains
them without bound: 200 repeated graph builds on 16 threads held 324 MB of
footprint where the same loop now plateaus at 156 MB, at the same wall time.
A mimalloc found through `TF_USE_SYSTEM_MIMALLOC` needs
`MI_ENABLE_LARGE_PAGES=0` for the same reason.

## trueform v0.10.2

The OBJ readers are parallel — all of them. The general reader drops from
44.5 ms to 6.5 ms on a million-triangle dragon, at parity with the
fixed-arity reader; the complete reader (normals, uvs, groups) from 76.3 ms
to 8.3 ms, and it can now state its arity — `read_obj<3>(path, tf::complete)`
— shedding the face tokenization and the offset table when the faces are
known triangles. The readers are held to each other on one file by committed
fixtures, a dropped malformed face no longer shifts the spans of the faces
after it, and the path overloads memory-map, so the file must not change
during the read.

`make_cdt` grew a config and a labels read. `tf::cdt_config{split_constraints,
regions}` is implicitly constructible from either member, so every existing
call stands; `tf::return_region_labels` returns the whole triangulation with
a label per face — `nesting` (the even-odd parity of region walls) or
`components` (the id of each wall-cut region, 0 the hull exterior) — region
zero included, because a map with the exterior erased cannot tell a hole from
an absence. Python: `tf.cdt(..., region_labels="nesting"|"components")`.

The Python layer catches up to the engine. `CsgGraph` takes one mesh (its
self arrangement) and takes dynamic meshes; the pairwise booleans are now
three expressions over the graph — byte-identical results, one compiled
pipeline, a smaller wheel that builds faster than before the feature
existed. New entries: `fit_similarity_alignment`, `euler_characteristic`,
`signed_distance` (single point or a batched `tf.Point`, negative inside),
and `graph.outer_shell()`, the shell read that no longer rebuilds a second
arrangement when you already hold the graph. `shape_index` keeps its
documented [-1, 1] contract — it previously escaped toward ±2 — and the
cdt entry normalizes non-contiguous inputs instead of misreading them.

Bundle containment is now decided by theorems instead of vertices. The
classification's parity cast seeds from a triangle's interior — stated
exactly, as the corner sum over its denominator, never materialized — so
a seed can no longer sit on another carrier and the answer no longer
depends on input face order. A cut face states its domain transition
like a whole one, so a body floating in the overlap of two solids is
contained by the overlap, not double-counted; and ray parity carries its
far term, so a region a sheet leaves unbounded contains exactly what it
should.

The three connected-component rules — manifold-edge, any-edge, vertex — now
carry hand-built fixtures pinning their exact label partitions, and their
docs state each rule's carrier.

## trueform v0.10.1

Repairs; no API breaks. `orient_faces_consistently` and
`ensure_positive_orientation` now return whether the surface is
orientable: every reversal is decided against the input winding and
applied after the walk, so one call settles every orientable
manifold-edge component, and a non-orientable component is left
untouched rather than half-repaired. An integral coordinate type votes
by face count, its lattice not holding a squared area.
`euler_characteristic` counts boundary edges correctly — a disk is 1,
an uncapped tube 0. `fit_similarity_alignment` measures the source
spread in the same space as its covariance, so a scaled frame no longer
skews the recovered scale. The VTK wrappers delegate orientation to the
core and surface its verdict.

## trueform v0.10.0 — one engine

Real-time geometric processing. Easy to use, robust on real-world data. C++,
Python, and TypeScript/WASM. **[Try it live](https://trueform.polydera.com/live-examples/boolean)**.

### The engine

**Every boolean runs on the CSG engine.** `tf::make_boolean` builds the same
N-form arrangement that powers `tf::make_csg_graph` and evaluates the
operation as a boolean expression over it. One arrangement answers any number
of boolean expressions, domain reads, and curve extractions; there is one
pipeline, one triangulator, one classification tier. On a 1000-pair
real-world corpus the engine matches 0.9 on the median pair with a
single-digit-percent p90 tail — while resolving coincidence exactly, taking
N-ary operands and sheets, and answering any number of reads from one build.
On an eleven-solid geological decomposition of 23 million triangles — the
N-ary shape at its heaviest, with thousands of faces pooling onto shared
planes — the engine is a third faster than 0.9: a pooled carrier's per-member
bookkeeping is stored as the sparse relation it is, so a plane carrying
40,000 coplanar faces costs its facts, not its dimensions.

**The arrangement is two tiers.** The local arrangement is the identity
tier: polygon intersections are classified exactly, and every crossing,
landing, and coincidence names one created identity on an exact integer
lattice — a point's identity is a canonical name, not a coordinate.
Coplanar faces, within a mesh and across meshes, pool onto one geometric
plane, so one plane is one identity and coincident walls cannot disagree.
The plane arrangement is the triangulation tier: each plane carrier
triangulates against its own constraint set, and a statement that cannot
stand as given is resolved — split at its crossings, welded at its
coincidences — in recovery waves that repeat until nothing new is stated.
A wave round's cost is proportional to what changed; an unchanged carrier
keeps its triangulation verbatim. Classification — components, fences,
radial fans, domains — reads the finished arrangement, and every product
(boolean meshes, domain cells, the outer shell, intersection curves) is a
read.

**Coincidence is resolved at the root.** When quantization collapses
geometry — duplicate vertices, sub-epsilon edges, folded slivers, vertices
landing on the intersection curve — the arrangement resolves each such
coincidence to a single created identity, and the surrounding faces re-emit
conformed to it. Raw arrangements are watertight by construction at these
sites.

**`tf::triangulated` is the same engine with the face as the carrier.** The
whole mesh is one world: every face triangulates on its own boundary, a
shared edge is one identity in both faces, and a self-crossing or folded
face is resolved — split at its own crossings, its coincident vertices
welded — never refused silently or dropped. A resolved face's positions are
unchanged; only its names are. The one entry takes an indexed mesh, a single
polygon, or a polygon soup — a soup is cleaned to shared-vertex identity
first, so the machinery below never sees one — and `Index` names the output
width: an `int64` mesh whose output fits `int32` asks for `int32` and gets
it. On quad meshes the exact engine outruns a float ear-cut kernel from
~90k faces; all-triangle meshes pass at copy speed plus an exact degeneracy
check.

**Refusals are a surface, not a silence.** `tf::return_refused` on
`triangulated`, `embedded_isocurves`, and `make_isobands` returns the input
faces a build declined, and `failed()` on a built graph names the carriers
still refusing after recovery. Nothing is dropped without a name, and the
surface costs zero when nothing refuses.

### Tolerance is a statement about the input

**A tolerance is the pitch the input's planes are quantized to; it never
widens a predicate.** One converter is built over the union of the input
forms; every face's plane — direction and offset — is rounded onto a grid
of that pitch, so two faces whose planes differ by less than it share one
plane. Every original vertex is then placed, at most the tolerance away,
onto the quantized planes its own faces stand on — the meet of three where
it is a corner, a line of two where it lies on a crease, its own tangent
plane where the surface is smooth. The result is the exact arrangement of
that moved mesh: every predicate below the placement runs exact, and a
weld is an identity, never a proximity. A tolerance of zero is the
identity — nothing moves, no placement table is built, and the result is
byte-identical to the exact arrangement of the input as given.

A wall doubled at less than the pitch becomes one wall; one doubled at
more stays two. A vertex a weld retires has no output point of its own —
the index map reports it absent. Plane identity is quantized-name
equality, so results from calls with different converter domains are not
identity-comparable; a session that needs one perturbation across calls
pins one converter.

### New: provenance selection

One arrangement, any read. `tf::csg::selection` names which forms' faces
reach the output, orthogonally to the boolean algebra:

```cpp
auto graph = tf::make_csg_graph(forms);              // A=0, B=1, C=2
auto e = (tf::csg::op(0) | tf::csg::op(1)) - tf::csg::op(2);

auto solid = tf::make_csg_mesh(graph, e);                          // as always
auto partA = tf::make_csg_mesh(graph, tf::csg::selection(0, e));   // A's walls of it
auto embed = tf::make_csg_mesh(graph, tf::csg::selection({0}));    // A cut by all, no boolean
```

The per-form parts partition the result face-for-face and re-weld into
the solid. `make_csg_domains` takes the same selection — each cell's
walls by contributor. Python: `graph.mesh(expr, selection=[0])`;
TypeScript: `graph.mesh(expr, { selection: [0] })`. Every existing
expression call is unchanged.

### The module map (breaking)

The ground the cut module held is split by mechanism:

- **`<trueform/arrangement.hpp>`** — mesh, polygon, and segment
  arrangements: the arrangement of a set of forms and its reads.
- **`<trueform/iso.hpp>`** — the scalar-field tier: `embedded_isocurves`,
  `make_isobands`, `make_isocontours`, `scalar_field_intersections`.
- **`<trueform/csg.hpp>`** — booleans, the CSG graph, expressions,
  selections, domains, and the outer shell.

Python and TypeScript follow the same map, so all three languages speak the
same module names; public function names, signatures, and defaults are
unchanged (TypeScript renames two result types:
`LabeledCutResult`/`LabeledCutResultWithCurves` are now
`LabeledBooleanResult`/`LabeledBooleanResultWithCurves`). There is no
compatibility shim. The removed entry points and their replacements:

**C++**

| Removed | Use instead |
|---------|-------------|
| `tf::embedded_intersection_curves` | `tf::make_mesh_arrangements` |
| `tf::embedded_self_intersection_curves` | `tf::make_polygon_arrangements` |
| `tf::intersections_between_polygons` | `tf::polygon_intersections` |
| `tf::intersections_within_polygons` | `tf::polygon_intersections` |
| `tf::triangulated_faces` | `tf::triangulated` |
| `tf::ear_cutter` | `tf::constrained_delaunay_triangulator`, or `tf::triangulated` for a whole mesh |
| `tf::delaunay_triangulator` | `tf::unconstrained_delaunay_triangulator`, or `tf::make_cdt(points)` |
| `tf::delaunay_flipper` | — no replacement; it existed only for the batch triangulator |

**Python**

| Removed | Use instead |
|---------|-------------|
| `tf.embedded_intersection_curves` | `tf.mesh_arrangements(..., return_curves=True)` |
| `tf.embedded_self_intersection_curves` | `tf.polygon_arrangements(..., return_curves=True)` |

**TypeScript**

| Removed | Use instead |
|---------|-------------|
| `embeddedIntersectionCurves` | `meshArrangements(meshes, { returnCurves: true })` |
| `embeddedSelfIntersectionCurves` | `polygonArrangements(mesh, { returnCurves: true })` |
| type `CutResultWithCurves` | `MeshArrangementResultWithCurves` / `PolygonArrangementResultWithCurves` |
| type `OuterShellOptions` | — `outerShell(mesh)` takes no options; the shell is a structural read of a settled arrangement |

### API changes

- `make_boolean(a, b, op[, sheets][, config])` — operands listed in `sheets`
  bound no volume and cut as oriented separators. An **open operand is a
  volume unless declared a sheet**; declare it a sheet for the previous
  open-mesh behavior.
- Booleans take `tf::arrangement_config` (an `intersect_config`, a
  `triangulation_type`, or both); `boolean_config` does not exist.
  Multi-nesting is always on.
- `make_boolean(..., tf::return_index_map)` returns `tf::stitch_index_map` —
  a producer-agnostic carrier consumed by `tf::stitch_*`.
- A swapped difference does not reverse winding.
- `intersect_config::tolerance` is the distance an input vertex may move to
  reach the lattice — a statement about the input, exact below it. A band on
  the predicates' comparisons does not exist.
- `tf::csg::inside(tags, e)` reads the named forms' surface lying INSIDE the
  region `e` bounds (both sides of a piece in it, stored winding) — how a
  sheet's surface within a solid is asked for; `tf::csg::selection` reads the
  region's boundary as before. `make_csg_domains` takes a boundary selection
  only.
- `tf::intersections_within_segments::build(segments)` takes no
  `intersect_config`: the segment tier is exact, like every tier.
- `make_intersection_curves` defaults to `primitives` in all three
  languages.
- `triangulated` resolves rather than refuses: a self-crossing face is
  split at its crossings, coincident vertices weld into one identity at the
  same coordinate — a resolved face need not name every vertex it was
  given; positions are unchanged, only names. Region fill is `labels != 0`.
- `tf::triangulated_faces` is removed. Its contract — corners indexed
  against a point table it did not return — cannot survive a resolution
  that mints a point; `tf::triangulated` returns the whole mesh, and the
  faces-level need is served by the arrangement mesh tier.
- The exact lattice is never a hardcoded default: entries and the
  triangulator classes resolve it from the coordinate type (float → int32,
  double → int64), and the planar-graph classes — which carry no coordinate
  type — take it from the caller. The iso cut resolves the same way,
  matching the arrangements.

- `fit_similarity_alignment` recovers the scale independent of the point
  count; it was `n` times too small
  ([#21](https://github.com/polydera/trueform/issues/21)).

### Validation

Full C++, Python, and TypeScript suites green; a 1000-pair real-world
corpus checked closed/manifold per pair against the recorded reference —
1000/1000 valid, including one pair the 0.9 reference itself failed; a
224-step iterated-carve chain; geological N-ary decomposition at
performance parity or better in every configuration.

The tolerance contract on the same corpus: at zero the output is
byte-identical to the exact arrangement, pair for pair; at bands of 1e-6
and 1e-5 every one of the 1000 arrangements is closed with no refusing
carrier and every boolean closed, where the retired mechanism left open
arrangements at both bands and annihilated half of the union mass at 1e-5.

**[Benchmarks](https://polydera.com/trueform/benchmarks)** — vs CGAL, VTK, libigl, Coal, FCL, nanoflann.

**[Documentation](https://trueform.polydera.com)** — tutorials, examples, and more.

### Installation

```bash
pip install trueform
python -m trueform.conan create
```

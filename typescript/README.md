# @polydera/trueform

Real-time geometric processing in TypeScript. WASM-backed NDArrays with vectorized numerical operations and spatial queries. Mesh booleans, registration, remeshing — at native speed in the browser and Node.js.

**[Documentation](https://trueform.polydera.com/ts/getting-started)** | **[Live Examples](https://trueform.polydera.com/live-examples/boolean)**

## Installation

```bash
npm install @polydera/trueform
```

## Requirements

- **Node.js** 18+ or any modern browser with WebAssembly support
- **SharedArrayBuffer** — required for multithreading. In browsers, the page must be served with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Node.js enables SharedArrayBuffer by default.

## Quick Tour

**NDArray** — WASM-backed numerical arrays:

```ts
import * as tf from "@polydera/trueform";

// Extract the z-column from mesh points — zero-copy view
const heights = mesh.points.take(null, 2);

// Two conditions: above ground AND close to a reference plane
const above = heights.gt(0);
const dists = tf.distance(mesh.points, plane);
const close = dists.lt(0.5);
const mask = above.and(close);

// Boolean-index to get matching points
const selected = mesh.points.booleanIndex(mask);
```

**Meshes** from NDArrays or files:

```ts
const faces = tf.ndarray(new Int32Array([0,1,2, 0,2,3, 0,3,1, 1,3,2]), [4, 3]);
const points = tf.ndarray(new Float32Array([
  0,0,0, 1,0,0, 0,1,0, 0,0,1
]), [4, 3]);
const mesh = tf.mesh(faces, points);

// Or read from file
const mesh = tf.readStl(await (await fetch("model.stl")).arrayBuffer());
```

**Spatial queries** — same functions for primitives, batches, and forms:

```ts
const plane = tf.plane(tf.vector(0, 0, 1), 0);
const scalars = tf.distance(pts, plane);            // NDArray [1000]

const { elementIds, distances } = tf.neighborSearch(mesh, segs);
const { hit, t } = tf.rayCast(ray, triangle);

// Batched — cast N rays against a mesh in one call
const { hits, ts, elementIds } = tf.rayCast(rays, mesh);
```

**Boolean operations:**

```ts
const union = tf.booleanUnion(mesh0, mesh1);

// With intersection curves
const diff = tf.booleanDifference(box, cylinder, { returnCurves: true });
// diff.mesh, diff.labels, diff.curves
```

**Mesh analysis:**

```ts
const { nComponents, labels } = tf.connectedComponents(mesh, "manifoldEdge");
const components = tf.splitIntoComponents(mesh, labels);
const { k0, k1, d0, d1 } = tf.principalDirections(mesh);
```

**Async** — every operation available off the main thread:

```ts
const d = await tf.async.distance(mesh, point);
const union = await tf.async.booleanUnion(mesh0, mesh1);
const { hits, ts } = await tf.async.rayCast(rays, mesh);
```

> [Full documentation](https://trueform.polydera.com/ts/modules) covers topology, isocontours, remeshing, registration, and more.

## Examples

- **[Guided Examples](https://trueform.polydera.com/ts/examples)** — Step-by-step walkthroughs for spatial queries, topology, and booleans
- **[Live Examples](https://trueform.polydera.com/live-examples/boolean)** — Interactive demos running in your browser

Run examples locally:

```bash
git clone https://github.com/polydera/trueform.git
cd trueform/typescript/examples
node core.mjs
```

## Benchmarks

| Operation | Input | Time | Speedup | Baseline |
|-----------|-------|------|---------|----------|
| Boolean Union | 2 × 1M | 28 ms | **84×** | CGAL `Simple_cartesian<double>` |
| Mesh–Mesh Curves | 2 × 1M | 7 ms | **233×** | CGAL `Simple_cartesian<double>` |
| ICP Registration | 1M | 7.7 ms | **93×** | libigl |
| Self-Intersection | 1M | 78 ms | **37×** | libigl EPECK (GMP/MPFR) |
| Isocontours | 1M, 16 cuts | 3.8 ms | **38×** | VTK `vtkContourFilter` |
| Connected Components | 1M | 15 ms | **10×** | CGAL |
| Boundary Paths | 1M | 12 ms | **11×** | CGAL |
| k-NN Query | 500K | 1.7 µs | **3×** | nanoflann k-d tree |
| Mesh–Mesh Distance | 2 × 1M | 0.2 ms | **2×** | Coal (FCL) `OBBRSS` |
| Decimation (50%) | 1M | 72 ms | **50×** | CGAL `edge_collapse` |
| Principal Curvatures | 1M | 25 ms | **55×** | libigl |

Apple M4 Max, 16 threads, Clang `-O3`. [Full methodology](https://trueform.polydera.com/ts/benchmarks)

## License

Dual-licensed: [PolyForm Noncommercial 1.0.0](https://github.com/polydera/trueform/blob/main/LICENSE.noncommercial) for noncommercial use, [commercial licenses](mailto:info@polydera.com) available.

## Contributing

See [CONTRIBUTING.md](https://github.com/polydera/trueform/blob/main/CONTRIBUTING.md) and [open issues](https://github.com/polydera/trueform/issues).

## Citation

```bibtex
@software{trueform2025,
    title={trueform: Real-time Geometric Processing},
    author={Sajovic, {\v{Z}}iga and {et al.}},
    organization={XLAB d.o.o.},
    year={2025},
    url={https://github.com/polydera/trueform}
}
```

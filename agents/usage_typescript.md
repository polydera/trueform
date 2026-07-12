# TypeScript Usage Patterns

How to USE trueform from TypeScript. Every pattern below is from the official documentation.

```ts
import * as tf from "@polydera/trueform";
await tf.init();  // Must be called before any operations
```

---

## 1. NDArray

### Creation

```ts
const a = tf.ndarray([1, 2, 3, 4, 5, 6], [2, 3]);           // from JS array
const b = tf.ndarray(new Float32Array([1, 2, 3]), [1, 3]);   // from TypedArray
const z = tf.zeros("float32", [100, 3]);
const o = tf.ones("int32", [10]);
const f = tf.full("float32", [4, 4], 0.5);
const I = tf.eye("float32", 4);
const r = tf.arange("int32", 10);                             // [0..9]
const l = tf.linspace(0, 1, 11);
const rnd = tf.random("float32", [100, 3], -1, 1);
```

### Properties

```ts
a.data;     // Float32Array | Int32Array | Int8Array (zero-copy view)
a.shape;    // number[] — settable
a.length;   // total elements
a.ndim;     // number of dimensions
a.dtype;    // "float32" | "int32" | "int8" | "bool"
```

### Element-wise Operations (broadcasting follows NumPy rules)

```ts
a.add(b);  a.add(1.0);  a.add_(b);   // in-place with _
a.sub(b);  a.mul(b);    a.div(b);    a.mod(b);
a.clip(0, 1);  a.clip_(0, 1);
A.matMul(B);                          // matrix multiply with broadcast
```

### Relational and Logical

```ts
a.gt(0);  a.lt(b);  a.eq(b);  a.ge(0);  a.le(b);  a.ne(0);
mask_a.and(mask_b);  mask_a.or(mask_b);  mask_a.not();
tf.where(mask, a, b);                 // conditional selection
```

### Reductions

```ts
pts.sum();       pts.sum(0);          // scalar or per-axis
pts.norm(1);                           // per-row L2 norm
pts.min();       pts.max();            // global
pts.argmin();    pts.argmax(0);        // index of min/max
mask.any();      mask.all();           // boolean reductions
mask.any(1);                           // per-row
```

### Indexing (zero-copy where possible)

```ts
pts.row(0);                            // single row
pts.slice(0, 10);                      // first 10 rows
pts.take(ids);                         // gather by index array
pts.take(null, 2);                     // column 2 (all rows)
pts.take(null, [0, 2]);               // columns 0 and 2
pts.booleanIndex(mask);                // rows where mask is true
```

### Shape, Sort, Combine

```ts
a.flatten();  a.reshape([50, 6]);  a.T;  a.transpose([1, 0]);
a.unsqueeze(0);  a.squeeze();
a.sort();  a.argsort();               // copies
a.sort_();                             // in-place
tf.unique(a);  tf.setUnion(a, b);     // set operations (sorted input)
tf.stack([a, b], 0);                   // new axis
tf.concatenate([a, b], 0);            // existing axis
a.clone();                             // deep copy
a.as("int32");                         // type cast
```

### Assignment (in-place, broadcasting)

```ts
pts.assign(1.0);                       // fill all
pts.assign(row);                       // broadcast [3] → all rows
pts.assign(mask, 0.0);                // zero where mask is true
pts.assign(ids, row);                  // set specific rows
```

---

## 2. Primitives (Typed NDArrays)

```ts
// Single
const p = tf.point(1, 2, 3);                                    // [3]
const v = tf.vector(0, 0, 1);                                   // [3]
const seg = tf.segment(tf.point(0,0,0), tf.point(1,1,1));      // [2, 3]
const ray = tf.ray(tf.point(0,0,0), tf.vector(0,0,1));         // [2, 3]
const tri = tf.triangle(tf.point(0,0,0), tf.point(1,0,0), tf.point(0,1,0));
const pl = tf.plane(tf.vector(0,0,1), 5.0);                    // [4]
const box = tf.aabb(tf.point(0,0,0), tf.point(1,1,1));         // [2, 3]

// Batch (leading dimension)
const pts = tf.point(new Float32Array(300), 3);                 // [100, 3]
const pts = tf.point(tf.random("float32", [50, 3]));           // [50, 3]
const rays = tf.ray(origins, directions);                       // [N, 2, 3]

// Properties
p.type;      // "point"
p.isBatch;   // false
p.count;     // 1
pts.count;   // 100
pts.at(0);   // single Point from batch
```

---

## 3. Mesh

```ts
// Create
const faces = tf.ndarray(new Int32Array([0,1,2, 0,2,3]), [2, 3]);
const points = tf.ndarray(new Float32Array([0,0,0, 1,0,0, 0,1,0, 1,1,0]), [4, 3]);
const mesh = tf.mesh(faces, points);

// From file
const mesh = tf.readStl(await response.arrayBuffer());

// Properties
mesh.numberOfFaces;  mesh.numberOfPoints;
mesh.faces;          // NDArrayInt32
mesh.points;         // NDArrayFloat32
```

### Topology (Lazy, Cached)

Topology structures are computed on first access and cached. If you modify faces or points, the cache invalidates and recomputes on next access. No manual rebuild needed.

```ts
mesh.faceMembership;    // OffsetBlockedBuffer — computed on first access, cached
mesh.manifoldEdgeLink;  // NDArrayInt32 — cached
mesh.faceLink;          // OffsetBlockedBuffer — cached
mesh.vertexLink;        // OffsetBlockedBuffer — cached
mesh.normals;           // NDArrayFloat32 (face normals) — cached
mesh.pointNormals;      // NDArrayFloat32 (vertex normals) �� cached

// Tree is also lazy — built on first spatial query or explicit call
mesh.buildTree();
```

### Transformations

Transformations are applied at query time — the underlying data stays in local coordinates.

```ts
mesh.transformation = tf.makeTranslation(5, 0, 0);
mesh.transformation = tf.makeRotation(90, "z");
mesh.transformation = null;  // clear
```

### Shallow copies (posed views)

`mesh.shallowCopy()` returns a new handle that **inherits everything
from the original** — same buffers, same cached tree, same topology —
and clears the transformation. Siblings diverge only when you
**reassign** data: `copy.points = …` or `copy.faces = …` affects that
handle only and invalidates only that handle's caches. Same on
`PointCloud`.

```ts
const base = tf.mesh(faces, points);
base.buildTree();

const viewA = base.shallowCopy();
viewA.transformation = tf.makeTranslation(5, 0, 0);
const viewB = base.shallowCopy();
viewB.transformation = tf.makeTranslation(-5, 0, 0);
tf.distance(viewA, viewB);                // both reuse base's tree

const moved = base.shallowCopy();
moved.points = translatedPoints;          // only `moved` diverges
tf.distance(moved, probe);                // rebuilds tree for `moved`
tf.distance(base, probe);                 // base's tree unchanged
```

### Memory Management

WASM objects are reference-counted. `FinalizationRegistry` handles cleanup when the JS object is garbage-collected, but GC timing is unpredictable. **Call `.delete()` explicitly when you're done** to free WASM memory immediately — especially in loops or long-running code.

```ts
const result = tf.booleanUnion(mesh0, mesh1);
// Use result...
result.faceLabels.delete();
result.labels.delete();
result.mesh.delete();

// Or use scope-based cleanup (TypeScript 5.2+)
using m = tf.mesh(faces, points);
// m.delete() called automatically at scope end
```

---

## 4. PointCloud and Curves

```ts
const pc = tf.pointCloud(tf.random("float32", [1000, 3]));
const pc = tf.pointCloud(mesh);  // from mesh points

const curves = tf.curves(paths_obb, points);
curves.length;  curves.paths;  curves.points;
```

---

## 5. Spatial Queries

```ts
// Distance
tf.distance(mesh, tf.point(0, 0, 0));                     // number
tf.distance(batchPoints, mesh);                             // NDArrayFloat32 [N]
tf.distance(meshA, meshB);                                  // number

// Closest point
const r = tf.closestPoint(tf.point(0,2,0), segment);
r.point;  r.distance2;

// Neighbor search
const r = tf.neighborSearch(mesh, tf.point(0.5, 0.5, 1));
r.elementId;  r.point;  r.distance2;

// Batch neighbor search
const rb = tf.neighborSearch(mesh, batchPoints);
rb.elementIds;  rb.points;  rb.distances;                  // NDArrays

// kNN
const r = tf.neighborSearch(mesh, point, { k: 5 });
r.elementIds;  r.points;  r.distances;                     // shape [5, ...]

// Batch kNN
const r = tf.neighborSearch(mesh, batchPoints, { k: 3 });
r.elementIds.shape;  // [N, 3]
r.counts;            // NDArrayInt32 [N] — actual count per query

// With radius
const r = tf.neighborSearch(mesh, point, { k: 10, radius: 2.0 });

// Ray casting
const r = tf.rayCast(ray, mesh);
if (r.hit) { r.elementId; r.t; }

// Batch rays
const rb = tf.rayCast(batchRays, mesh);
rb.hits;  rb.ts;  rb.elementIds;                           // NDArrays

// Per-ray bounds
tf.rayCast(rays, mesh, { minT: minTs, maxT: maxTs });

// Intersection test
tf.intersects(meshA, meshB);                                // boolean
tf.intersects(mesh, batchSegments);                         // NDArrayBool
```

---

## 6. Boolean Operations

```ts
// Union / Intersection / Difference
const { mesh, labels, faceLabels } = tf.booleanUnion(mesh0, mesh1);
const { mesh, labels, faceLabels } = tf.booleanIntersection(mesh0, mesh1);
const { mesh, labels, faceLabels } = tf.booleanDifference(mesh0, mesh1);

// With intersection curves
const { mesh, labels, faceLabels, curves } = tf.booleanUnion(mesh0, mesh1, {
  returnCurves: true,
});

// Mesh arrangements (N inputs)
const { mesh, tagLabels, faceLabels } = tf.meshArrangements([mesh0, mesh1, mesh2]);
const { mesh, tagLabels, faceLabels, curves } = tf.meshArrangements(
  [mesh0, mesh1], { returnCurves: true });

// Embedded intersection curves
const { mesh, faceLabels } = tf.embeddedIntersectionCurves(mesh0, mesh1);
const { mesh, faceLabels } = tf.embeddedSelfIntersectionCurves(mesh);

// Isobands
const { mesh, labels, faceLabels } = tf.isobands(mesh, scalars, cutValues);
const { mesh, labels, faceLabels, curves } = tf.isobands(mesh, scalars, cutValues, {
  returnCurves: true, selectedBands: [1, 3],
});
```

### CSG graph: build once, query many (N-ary)

```typescript
const graph = tf.csgGraph([a, b, c], { triangulation: "refinedCdt" });
// or off-thread: await tf.async.csgGraph([a, b, c])

const diff = graph.mesh(tf.op(0).sub(tf.op(1)));       // .or/.and/.sub/.not
const full = graph.mesh();                              // full arrangement mesh
const lab  = graph.mesh(tf.op(0).or(1), { returnSourceIds: true });
const doms = graph.domains({ returnIndexMap: true });   // opts in the expr slot
const seams = graph.intersectionCurves();
graph.delete();                                         // explicit lifetime
```


---

## 7. Mesh Analysis

```ts
const { nComponents, labels } = tf.connectedComponents(mesh, "manifoldEdge");
const components = tf.splitIntoComponents(mesh, labels);

const bp = tf.boundaryPaths(mesh);
const { k0, k1, d0, d1 } = tf.principalDirections(mesh);
const si = tf.shapeIndex(mesh);
```

---

## 8. Mesh Processing

```ts
const clean = tf.cleaned(mesh);
const oriented = tf.positivelyOriented(mesh);
const tri = tf.triangulate({ faces: quadFaces, points });

const decimated = tf.decimated(mesh, 0.5);
const mel = tf.meanEdgeLength(decimated);
const remeshed = tf.isotropicRemeshed(decimated, mel);
```

---

## 9. Registration

```ts
const { transformation, error } = tf.fitIcpAlignment(source, target);
const { transformation } = tf.fitRigidAlignment(source, target);
const { transformation } = tf.fitObbAlignment(source, target);
const error = tf.chamferError(source, target);
```

---

## 10. I/O

```ts
const mesh = tf.readStl(arrayBuffer);
const mesh = tf.readObj(arrayBuffer);
const stlBuffer = tf.writeStl(mesh);
const objBuffer = tf.writeObj(mesh);
```

---

## 11. Async (Non-Blocking)

Every operation is available as `tf.async.*`:

```ts
const d = await tf.async.distance(mesh, point);
const { mesh, labels, faceLabels } = await tf.async.booleanUnion(mesh0, mesh1);
const { hits, ts } = await tf.async.rayCast(rays, mesh);
const r = await tf.async.neighborSearch(mesh, points, { k: 5 });
await tf.async.sort(arr);
await tf.async.sum(pts, 0);
```

---

## 12. Transformations

```ts
tf.makeTranslation(1, 0, 0);
tf.makeTranslation([1, 0, 0]);
tf.makeRotation(90, "z");                              // degrees, axis name
tf.makeRotation(45, [1, 0, 0], [0, 0, 5]);           // degrees, axis, pivot
tf.makeRandomRotation();
tf.makeRandomRotation(centroid);

// Three.js interop (column-major ↔ row-major)
const tfMat = tf.ndarray(threeMatrix4.elements, [4, 4]).T;
const m4 = new THREE.Matrix4().fromArray(tfMat.T.data);
```

---

## 13. OffsetBlockedBuffer (Variable-Length Blocks)

```ts
const obb = tf.offsetBlockedBuffer(offsets, data);
obb.length;           // number of blocks
obb.get(0);           // NDArray for block 0
obb.offsets;          // NDArrayInt32
obb.data;             // NDArrayInt32
for (const block of obb) { block.data; }
```

---

## 14. Memory Management

```ts
// Reference-counted. FinalizationRegistry handles auto-cleanup.
// Optional explicit cleanup:
mesh.delete();                  // immediate
result.mesh.delete();
result.labels.delete();

// Scope-based (TypeScript 5.2+ using declaration)
using m = tf.mesh(faces, points);
// m.delete() called automatically at scope end
```

import { describe, test, log, assert, getTf } from "./harness.mjs";

// A grid of triangles: n x n quads, two triangles each.
function gridArrays(n) {
  const pts = new Float32Array((n + 1) * (n + 1) * 3);
  for (let j = 0; j <= n; j++)
    for (let i = 0; i <= n; i++) {
      const k = (j * (n + 1) + i) * 3;
      pts[k] = i; pts[k + 1] = j; pts[k + 2] = 0;
    }
  const faces = new Int32Array(n * n * 6);
  let f = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
      faces.set([a, b, d, a, d, c], f);
      f += 6;
    }
  return { faces, pts };
}

describe("Handle lifetime", () => {

  // ==========================================================================
  test("mesh / connectedComponents / pointCloud loop does not grow the heap", () => {
    const tf = getTf();
    const { faces, pts } = gridArrays(200);
    const meshBytes = faces.byteLength + pts.byteLength;
    const heap = () => tf.native().wasmMemory.buffer.byteLength;

    const once = () => {
      const m = tf.mesh(faces, pts);
      for (let k = 0; k < 8; k++) {
        const r = tf.connectedComponents(m, "edge");
        r.labels.delete();
      }
      const pc = tf.pointCloud(m);
      pc.delete();
      m.delete();
    };

    for (let i = 0; i < 5; i++) once();
    const before = heap();
    for (let i = 0; i < 200; i++) once();
    const grown = heap() - before;
    log(`  heap grew ${grown} bytes over 200 iterations (mesh ${meshBytes} bytes)`, "line-pass");
    console.log(`handle-lifetime: heap grew ${grown} bytes over 200 iterations, one mesh = ${meshBytes} bytes`);
    assert(grown < meshBytes, `heap grew ${grown} bytes, one mesh is ${meshBytes}`);
  });

  // ==========================================================================
  test("a getter's handle disposed early leaves the mesh's own share standing", () => {
    const tf = getTf();
    const { faces, pts } = gridArrays(8);
    const m = tf.mesh(faces, pts);

    const link = m.faceLink;
    const n = link.length;
    link.delete();
    const again = m.faceLink;
    assert(again.length === n, `faceLink length ${again.length} after early dispose, expected ${n}`);
    again.delete();

    const f = m.faces;
    f.delete();
    assert(m.numberOfFaces === faces.length / 3, "numberOfFaces after disposing faces getter");
    const p = m.points;
    assert(p.data.length === pts.length, "points still readable after disposing an earlier getter");
    p.delete();

    const r = tf.connectedComponents(m, "edge");
    assert(r.nComponents === 1, `grid is one component, got ${r.nComponents}`);
    r.labels.delete();

    const pc = tf.pointCloud(m);
    const pp = pc.points;
    assert(pp.shape[0] === pts.length / 3, "point cloud keeps its points");
    pp.delete();
    pc.delete();
    // the mesh outlives the cloud built from it
    assert(m.faceLink.length === n, "mesh link intact after the cloud is deleted");
    m.delete();
  });
});

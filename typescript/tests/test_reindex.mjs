import { describe, test, log, assert, getTf } from "./harness.mjs";

describe("Reindex", () => {

  // ==========================================================================
  test("reindexedByMask (keep half the faces)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const nf = sphere.numberOfFaces;

    // Mask: keep first half of faces
    const maskData = new Int8Array(nf);
    for (let i = 0; i < Math.floor(nf / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [nf]);

    const filtered = tf.reindexedByMask(sphere, mask);
    assert(filtered.numberOfFaces === Math.floor(nf / 2),
      `expected ${Math.floor(nf / 2)} faces, got ${filtered.numberOfFaces}`);
    assert(filtered.numberOfPoints <= sphere.numberOfPoints,
      "filtered should have <= original points");
    assert(filtered.numberOfPoints > 0, "filtered should have > 0 points");
    log(`  ${nf} → ${filtered.numberOfFaces} faces, ${filtered.numberOfPoints} points`, "line-pass");

    filtered.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexedByMask with returnIndexMap", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const nf = sphere.numberOfFaces;

    const maskData = new Int8Array(nf);
    for (let i = 0; i < Math.floor(nf / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [nf]);

    const result = tf.reindexedByMask(sphere, mask, { returnIndexMap: true });
    assert(result.mesh.numberOfFaces === Math.floor(nf / 2),
      `expected ${Math.floor(nf / 2)} faces, got ${result.mesh.numberOfFaces}`);
    assert(result.faceMap.keptIds.shape[0] === result.mesh.numberOfFaces,
      "faceMap.keptIds length should match filtered face count");
    assert(result.pointMap.keptIds.shape[0] === result.mesh.numberOfPoints,
      "pointMap.keptIds length should match filtered point count");
    log(`  faceMap: ${result.faceMap.keptIds.shape[0]} kept, pointMap: ${result.pointMap.keptIds.shape[0]} kept`, "line-pass");

    result.pointMap.delete();
    result.faceMap.delete();
    result.mesh.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexedByIds (extract specific faces)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    const ids = tf.ndarray(new Int32Array([0, 1, 2, 3, 4]), [5]);
    const filtered = tf.reindexedByIds(sphere, ids);
    assert(filtered.numberOfFaces === 5, `expected 5 faces, got ${filtered.numberOfFaces}`);
    assert(filtered.numberOfPoints > 0, "should have points");
    log(`  extracted 5 faces → ${filtered.numberOfPoints} points`, "line-pass");

    filtered.delete();
    ids.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexedByIds with returnIndexMap", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    const ids = tf.ndarray(new Int32Array([0, 1, 2]), [3]);
    const result = tf.reindexedByIds(sphere, ids, { returnIndexMap: true });
    assert(result.mesh.numberOfFaces === 3, `expected 3 faces, got ${result.mesh.numberOfFaces}`);
    assert(result.faceMap.keptIds.shape[0] === 3, "faceMap should have 3 kept IDs");
    log(`  3 faces, ${result.mesh.numberOfPoints} points`, "line-pass");

    result.pointMap.delete();
    result.faceMap.delete();
    result.mesh.delete();
    ids.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexedByMaskOnPoints (point mask → face filter)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const np = sphere.numberOfPoints;

    // Keep first half of points
    const maskData = new Int8Array(np);
    for (let i = 0; i < Math.floor(np / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [np]);

    const filtered = tf.reindexedByMaskOnPoints(sphere, mask);
    assert(filtered.numberOfFaces > 0, "should have some faces");
    assert(filtered.numberOfFaces < sphere.numberOfFaces,
      "should have fewer faces than original");
    assert(filtered.numberOfPoints <= Math.floor(np / 2),
      "should have at most kept-point-count points");
    log(`  ${sphere.numberOfFaces} → ${filtered.numberOfFaces} faces (point mask)`, "line-pass");

    filtered.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexedByIdsOnPoints (point IDs → face filter)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    // Keep first 20 point IDs
    const idsArr = new Int32Array(20);
    for (let i = 0; i < 20; i++) idsArr[i] = i;
    const ids = tf.ndarray(idsArr, [20]);

    const filtered = tf.reindexedByIdsOnPoints(sphere, ids);
    assert(filtered.numberOfFaces > 0, "should have some faces");
    assert(filtered.numberOfFaces < sphere.numberOfFaces,
      "should have fewer faces than original");
    log(`  kept 20 point IDs → ${filtered.numberOfFaces} faces`, "line-pass");

    filtered.delete();
    ids.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexed (apply existing index maps from reindexedByMask)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const nf = sphere.numberOfFaces;

    // Get index maps via reindexedByMask
    const maskData = new Int8Array(nf);
    for (let i = 0; i < Math.floor(nf / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [nf]);
    const r1 = tf.reindexedByMask(sphere, mask, { returnIndexMap: true });

    // Re-apply the same maps to the original mesh
    const r2 = tf.reindexed(sphere, r1.faceMap, r1.pointMap);
    assert(r2.numberOfFaces === r1.mesh.numberOfFaces,
      `expected ${r1.mesh.numberOfFaces} faces, got ${r2.numberOfFaces}`);
    assert(r2.numberOfPoints === r1.mesh.numberOfPoints,
      `expected ${r1.mesh.numberOfPoints} points, got ${r2.numberOfPoints}`);
    log(`  re-applied maps: ${r2.numberOfFaces} faces, ${r2.numberOfPoints} points`, "line-pass");

    r2.delete();
    r1.pointMap.delete();
    r1.faceMap.delete();
    r1.mesh.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexed (NDArray — gather by keptIds)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const nf = sphere.numberOfFaces;

    // Get an IndexMap
    const maskData = new Int8Array(nf);
    for (let i = 0; i < Math.floor(nf / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [nf]);
    const result = tf.reindexedByMask(sphere, mask, { returnIndexMap: true });

    // Apply to an NDArray (e.g., per-face data)
    const faceData = tf.arange("int32", 0, nf);
    const gathered = tf.reindexed(faceData, result.faceMap);
    assert(gathered.shape[0] === result.mesh.numberOfFaces,
      `expected ${result.mesh.numberOfFaces} elements, got ${gathered.shape[0]}`);
    log(`  gathered ${gathered.shape[0]} elements from NDArray`, "line-pass");

    gathered.delete();
    faceData.delete();
    result.pointMap.delete();
    result.faceMap.delete();
    result.mesh.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("concatenateMeshes (two spheres)", () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 8, 8);
    const s2 = tf.sphereMesh(0.5, 6, 6);

    const combined = tf.concatenateMeshes(s1, s2);
    assert(combined.numberOfFaces === s1.numberOfFaces + s2.numberOfFaces,
      `expected ${s1.numberOfFaces + s2.numberOfFaces} faces, got ${combined.numberOfFaces}`);
    assert(combined.numberOfPoints === s1.numberOfPoints + s2.numberOfPoints,
      `expected ${s1.numberOfPoints + s2.numberOfPoints} points, got ${combined.numberOfPoints}`);
    log(`  ${s1.numberOfFaces}+${s2.numberOfFaces} = ${combined.numberOfFaces} faces`, "line-pass");

    combined.delete();
    s2.delete();
    s1.delete();
  });

  // ==========================================================================
  test("concatenateMeshes (three meshes)", () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 6, 6);
    const s2 = tf.boxMesh(1, 1, 1);
    const s3 = tf.cylinderMesh(0.5, 2.0, 8);

    const combined = tf.concatenateMeshes(s1, s2, s3);
    const expectedFaces = s1.numberOfFaces + s2.numberOfFaces + s3.numberOfFaces;
    const expectedPoints = s1.numberOfPoints + s2.numberOfPoints + s3.numberOfPoints;
    assert(combined.numberOfFaces === expectedFaces,
      `expected ${expectedFaces} faces, got ${combined.numberOfFaces}`);
    assert(combined.numberOfPoints === expectedPoints,
      `expected ${expectedPoints} points, got ${combined.numberOfPoints}`);
    log(`  3 meshes → ${combined.numberOfFaces} faces, ${combined.numberOfPoints} points`, "line-pass");

    combined.delete();
    s3.delete();
    s2.delete();
    s1.delete();
  });

  // ==========================================================================
  test("splitIntoComponents (round-trip with connectedComponents)", () => {
    const tf = getTf();

    // Create two separate spheres, concatenate them
    const s1 = tf.sphereMesh(1.0, 8, 8);
    const s2 = tf.sphereMesh(0.5, 6, 6);
    const combined = tf.concatenateMeshes(s1, s2);

    // Label connected components
    const cc = tf.connectedComponents(combined, "edge");
    assert(cc.nComponents === 2, `expected 2 components, got ${cc.nComponents}`);

    // Split by those labels
    const split = tf.splitIntoComponents(combined, cc.labels);
    assert(split.components.length === 2, `expected 2 components, got ${split.components.length}`);
    assert(split.labels.length === 2, `expected 2 labels, got ${split.labels.length}`);

    // Component face counts should sum to combined
    const totalFaces = split.components[0].numberOfFaces + split.components[1].numberOfFaces;
    assert(totalFaces === combined.numberOfFaces,
      `component faces ${totalFaces} != combined ${combined.numberOfFaces}`);
    log(`  split into ${split.components.length} components: ${split.components[0].numberOfFaces} + ${split.components[1].numberOfFaces} faces`, "line-pass");

    for (const c of split.components) c.delete();
    cc.labels.delete();
    combined.delete();
    s2.delete();
    s1.delete();
  });

  // ==========================================================================
  test("splitIntoComponents (single component → 1 mesh)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 8, 8);

    const cc = tf.connectedComponents(sphere, "edge");
    assert(cc.nComponents === 1, `expected 1 component, got ${cc.nComponents}`);

    const split = tf.splitIntoComponents(sphere, cc.labels);
    assert(split.components.length === 1, `expected 1 component, got ${split.components.length}`);
    assert(split.components[0].numberOfFaces === sphere.numberOfFaces,
      "single component should have all faces");
    log(`  1 component: ${split.components[0].numberOfFaces} faces`, "line-pass");

    for (const c of split.components) c.delete();
    cc.labels.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("concatenateMeshes with transformation + connectedComponents", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 8, 8);
    const box = tf.boxMesh(1, 1, 1);

    // Translate the box far away so they don't overlap
    box.transformation = tf.makeTranslation(10, 0, 0);

    const combined = tf.concatenateMeshes(sphere, box);
    assert(combined.numberOfFaces === sphere.numberOfFaces + box.numberOfFaces,
      `expected ${sphere.numberOfFaces + box.numberOfFaces} faces, got ${combined.numberOfFaces}`);

    // Verify box points were actually transformed:
    // box is [-0.5,0.5]^3 translated by +10 in x → x column min should be >= 9.0
    const boxPts = combined.points.slice(sphere.numberOfPoints);  // [boxN, 3]
    const colMins = boxPts.min(0);                                // [3] per-column min
    const boxMinX = colMins.get(0);
    assert(boxMinX >= 9.0, `box min x should be >= 9.0 after translation, got ${boxMinX}`);
    log(`  box min x after translation: ${boxMinX.toFixed(2)}`, "line-pass");

    const cc = tf.connectedComponents(combined, "edge");
    assert(cc.nComponents === 2, `expected 2 components, got ${cc.nComponents}`);
    log(`  sphere + translated box → ${cc.nComponents} components`, "line-pass");

    colMins.delete();
    boxPts.delete();
    cc.labels.delete();
    combined.delete();
    box.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("splitIntoComponents (per-face label counts match component face counts)", () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 8, 8);
    const s2 = tf.sphereMesh(0.5, 6, 6);
    const combined = tf.concatenateMeshes(s1, s2);

    const cc = tf.connectedComponents(combined, "edge");
    const split = tf.splitIntoComponents(combined, cc.labels);

    // Count per-face labels (from cc.labels, not split.labels)
    const count0 = tf.sum(cc.labels.eq(0));
    const count1 = tf.sum(cc.labels.eq(1));
    assert(count0 === split.components[0].numberOfFaces,
      `label 0 count ${count0} != component 0 faces ${split.components[0].numberOfFaces}`);
    assert(count1 === split.components[1].numberOfFaces,
      `label 1 count ${count1} != component 1 faces ${split.components[1].numberOfFaces}`);
    log(`  per-face label counts: ${count0} + ${count1} = ${combined.numberOfFaces}`, "line-pass");

    // split.labels is per-component unique label values
    assert(split.labels.length === 2, `expected 2 unique labels, got ${split.labels.length}`);

    for (const c of split.components) c.delete();
    split.labels.delete();
    cc.labels.delete();
    combined.delete();
    s2.delete();
    s1.delete();
  });

  // ==========================================================================
  test("splitIntoComponents (per-face label counts, async)", async () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 8, 8);
    const s2 = tf.sphereMesh(0.5, 6, 6);
    const combined = tf.concatenateMeshes(s1, s2);

    const cc = tf.connectedComponents(combined, "edge");
    const split = await tf.async.splitIntoComponents(combined, cc.labels);

    const count0 = tf.sum(cc.labels.eq(0));
    const count1 = tf.sum(cc.labels.eq(1));
    assert(count0 === split.components[0].numberOfFaces,
      `label 0 count ${count0} != component 0 faces ${split.components[0].numberOfFaces}`);
    assert(count1 === split.components[1].numberOfFaces,
      `label 1 count ${count1} != component 1 faces ${split.components[1].numberOfFaces}`);
    log(`  async: per-face label counts: ${count0} + ${count1} = ${combined.numberOfFaces}`, "line-pass");

    assert(split.labels.length === 2, `expected 2 unique labels, got ${split.labels.length}`);

    for (const c of split.components) c.delete();
    split.labels.delete();
    cc.labels.delete();
    combined.delete();
    s2.delete();
    s1.delete();
  });

  // ==========================================================================
  test("splitIntoComponents after boolean (sync)", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1.0, 10, 10);
    const s1 = tf.sphereMesh(1.0, 10, 10);
    s1.transformation = tf.makeTranslation(0.8, 0, 0);

    const { mesh, labels } = tf.booleanDifference(s0, s1);
    const split = tf.splitIntoComponents(mesh, labels);
    assert(split.components.length > 0, `expected >0 components, got ${split.components.length}`);

    const totalFaces = split.components.reduce((s, c) => s + c.numberOfFaces, 0);
    assert(totalFaces === mesh.numberOfFaces,
      `component faces ${totalFaces} != mesh faces ${mesh.numberOfFaces}`);
    log(`  boolean → split: ${split.components.length} components, ${totalFaces} faces`, "line-pass");

    for (const c of split.components) c.delete();
    split.labels.delete();
    labels.delete();
    mesh.delete();
    s1.delete();
    s0.delete();
  });

  // ==========================================================================
  test("splitIntoComponents after boolean (async)", async () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1.0, 10, 10);
    const s1 = tf.sphereMesh(1.0, 10, 10);
    s1.transformation = tf.makeTranslation(0.8, 0, 0);

    const { mesh, labels } = await tf.async.booleanDifference(s0, s1);
    const split = await tf.async.splitIntoComponents(mesh, labels);
    assert(split.components.length > 0, `expected >0 components, got ${split.components.length}`);

    const totalFaces = split.components.reduce((s, c) => s + c.numberOfFaces, 0);
    assert(totalFaces === mesh.numberOfFaces,
      `component faces ${totalFaces} != mesh faces ${mesh.numberOfFaces}`);
    log(`  async boolean → split: ${split.components.length} components, ${totalFaces} faces`, "line-pass");

    for (const c of split.components) c.delete();
    split.labels.delete();
    labels.delete();
    mesh.delete();
    s1.delete();
    s0.delete();
  });

  // ==========================================================================
  test("reindexedByMask → reindexed round-trip preserves geometry", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const nf = box.numberOfFaces;

    // Keep all faces via mask
    const allTrue = tf.ndarray(new Int8Array(nf).fill(1), [nf]);
    const result = tf.reindexedByMask(box, allTrue, { returnIndexMap: true });

    assert(result.mesh.numberOfFaces === nf,
      `expected ${nf} faces, got ${result.mesh.numberOfFaces}`);
    assert(result.mesh.numberOfPoints === box.numberOfPoints,
      `expected ${box.numberOfPoints} points, got ${result.mesh.numberOfPoints}`);
    log(`  all-true mask preserves ${nf} faces, ${box.numberOfPoints} points`, "line-pass");

    result.pointMap.delete();
    result.faceMap.delete();
    result.mesh.delete();
    allTrue.delete();
    box.delete();
  });

  // ==========================================================================
  test("async: reindexedByMask (keep half the faces)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const nf = sphere.numberOfFaces;

    const maskData = new Int8Array(nf);
    for (let i = 0; i < Math.floor(nf / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [nf]);

    const filtered = await tf.async.reindexedByMask(sphere, mask);
    assert(filtered.numberOfFaces === Math.floor(nf / 2),
      `expected ${Math.floor(nf / 2)} faces, got ${filtered.numberOfFaces}`);
    assert(filtered.numberOfPoints <= sphere.numberOfPoints,
      "filtered should have <= original points");
    assert(filtered.numberOfPoints > 0, "filtered should have > 0 points");
    log(`  async: ${nf} → ${filtered.numberOfFaces} faces, ${filtered.numberOfPoints} points`, "line-pass");

    filtered.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("async: reindexedByIds (extract specific faces)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    const ids = tf.ndarray(new Int32Array([0, 1, 2, 3, 4]), [5]);
    const filtered = await tf.async.reindexedByIds(sphere, ids);
    assert(filtered.numberOfFaces === 5, `expected 5 faces, got ${filtered.numberOfFaces}`);
    assert(filtered.numberOfPoints > 0, "should have points");
    log(`  async: extracted 5 faces → ${filtered.numberOfPoints} points`, "line-pass");

    filtered.delete();
    ids.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("async: reindexedByMaskOnPoints (point mask → face filter)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const np = sphere.numberOfPoints;

    const maskData = new Int8Array(np);
    for (let i = 0; i < Math.floor(np / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [np]);

    const filtered = await tf.async.reindexedByMaskOnPoints(sphere, mask);
    assert(filtered.numberOfFaces > 0, "should have some faces");
    assert(filtered.numberOfFaces < sphere.numberOfFaces,
      "should have fewer faces than original");
    assert(filtered.numberOfPoints <= Math.floor(np / 2),
      "should have at most kept-point-count points");
    log(`  async: ${sphere.numberOfFaces} → ${filtered.numberOfFaces} faces (point mask)`, "line-pass");

    filtered.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("async: reindexedByIdsOnPoints (point IDs → face filter)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    const idsArr = new Int32Array(20);
    for (let i = 0; i < 20; i++) idsArr[i] = i;
    const ids = tf.ndarray(idsArr, [20]);

    const filtered = await tf.async.reindexedByIdsOnPoints(sphere, ids);
    assert(filtered.numberOfFaces > 0, "should have some faces");
    assert(filtered.numberOfFaces < sphere.numberOfFaces,
      "should have fewer faces than original");
    log(`  async: kept 20 point IDs → ${filtered.numberOfFaces} faces`, "line-pass");

    filtered.delete();
    ids.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("async: concatenateMeshes (two spheres)", async () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 8, 8);
    const s2 = tf.sphereMesh(0.5, 6, 6);

    const combined = await tf.async.concatenateMeshes(s1, s2);
    assert(combined.numberOfFaces === s1.numberOfFaces + s2.numberOfFaces,
      `expected ${s1.numberOfFaces + s2.numberOfFaces} faces, got ${combined.numberOfFaces}`);
    assert(combined.numberOfPoints === s1.numberOfPoints + s2.numberOfPoints,
      `expected ${s1.numberOfPoints + s2.numberOfPoints} points, got ${combined.numberOfPoints}`);
    log(`  async: ${s1.numberOfFaces}+${s2.numberOfFaces} = ${combined.numberOfFaces} faces`, "line-pass");

    combined.delete();
    s2.delete();
    s1.delete();
  });

});

describe("Reindex (float64)", () => {

  // ==========================================================================
  test("reindexedByMask (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });
    const nf = sphere.numberOfFaces;

    const maskData = new Int8Array(nf);
    for (let i = 0; i < Math.floor(nf / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [nf]);

    const filtered = tf.reindexedByMask(sphere, mask);
    assert(filtered.dtype === "float64", "result dtype is float64");
    assert(filtered.numberOfFaces === Math.floor(nf / 2),
      `expected ${Math.floor(nf / 2)} faces, got ${filtered.numberOfFaces}`);
    assert(filtered.numberOfPoints > 0, "filtered should have > 0 points");
    log(`  reindexedByMask (f64): ${nf} → ${filtered.numberOfFaces} faces`, "line-pass");

    filtered.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexedByMask with returnIndexMap (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });
    const nf = sphere.numberOfFaces;

    const maskData = new Int8Array(nf);
    for (let i = 0; i < Math.floor(nf / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [nf]);

    const result = tf.reindexedByMask(sphere, mask, { returnIndexMap: true });
    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces === Math.floor(nf / 2),
      `expected ${Math.floor(nf / 2)} faces, got ${result.mesh.numberOfFaces}`);
    assert(result.faceMap.keptIds.shape[0] === result.mesh.numberOfFaces,
      "faceMap.keptIds length matches");
    assert(result.pointMap.keptIds.shape[0] === result.mesh.numberOfPoints,
      "pointMap.keptIds length matches");
    log(`  reindexedByMask + maps (f64): ${result.faceMap.keptIds.shape[0]} kept`, "line-pass");

    result.pointMap.delete();
    result.faceMap.delete();
    result.mesh.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexedByIds (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });

    const ids = tf.ndarray(new Int32Array([0, 1, 2, 3, 4]), [5]);
    const filtered = tf.reindexedByIds(sphere, ids);
    assert(filtered.dtype === "float64", "result dtype is float64");
    assert(filtered.numberOfFaces === 5, `expected 5 faces, got ${filtered.numberOfFaces}`);
    log(`  reindexedByIds (f64): 5 faces`, "line-pass");

    filtered.delete();
    ids.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexedByMaskOnPoints (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });
    const np = sphere.numberOfPoints;

    const maskData = new Int8Array(np);
    for (let i = 0; i < Math.floor(np / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [np]);

    const filtered = tf.reindexedByMaskOnPoints(sphere, mask);
    assert(filtered.dtype === "float64", "result dtype is float64");
    assert(filtered.numberOfFaces > 0, "should have some faces");
    log(`  reindexedByMaskOnPoints (f64): ${filtered.numberOfFaces} faces`, "line-pass");

    filtered.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexedByIdsOnPoints (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });

    const idsArr = new Int32Array(20);
    for (let i = 0; i < 20; i++) idsArr[i] = i;
    const ids = tf.ndarray(idsArr, [20]);

    const filtered = tf.reindexedByIdsOnPoints(sphere, ids);
    assert(filtered.dtype === "float64", "result dtype is float64");
    assert(filtered.numberOfFaces > 0, "should have some faces");
    log(`  reindexedByIdsOnPoints (f64): ${filtered.numberOfFaces} faces`, "line-pass");

    filtered.delete();
    ids.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("reindexed (apply index maps, float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });
    const nf = sphere.numberOfFaces;

    const maskData = new Int8Array(nf);
    for (let i = 0; i < Math.floor(nf / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [nf]);
    const r1 = tf.reindexedByMask(sphere, mask, { returnIndexMap: true });

    const r2 = tf.reindexed(sphere, r1.faceMap, r1.pointMap);
    assert(r2.dtype === "float64", "reindexed result dtype is float64");
    assert(r2.numberOfFaces === r1.mesh.numberOfFaces,
      `expected ${r1.mesh.numberOfFaces} faces, got ${r2.numberOfFaces}`);
    log(`  reindexed (f64): ${r2.numberOfFaces} faces`, "line-pass");

    r2.delete();
    r1.pointMap.delete();
    r1.faceMap.delete();
    r1.mesh.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("concatenateMeshes (float64)", () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 8, 8, { dtype: "float64" });
    const s2 = tf.sphereMesh(0.5, 6, 6, { dtype: "float64" });

    const combined = tf.concatenateMeshes(s1, s2);
    assert(combined.dtype === "float64", "result dtype is float64");
    assert(combined.numberOfFaces === s1.numberOfFaces + s2.numberOfFaces,
      `expected ${s1.numberOfFaces + s2.numberOfFaces} faces, got ${combined.numberOfFaces}`);
    log(`  concatenateMeshes (f64): ${combined.numberOfFaces} faces`, "line-pass");

    combined.delete();
    s2.delete();
    s1.delete();
  });

  // ==========================================================================
  test("splitIntoComponents (float64)", () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 8, 8, { dtype: "float64" });
    const s2 = tf.sphereMesh(0.5, 6, 6, { dtype: "float64" });
    const combined = tf.concatenateMeshes(s1, s2);

    const cc = tf.connectedComponents(combined, "edge");
    assert(cc.nComponents === 2, `expected 2 components, got ${cc.nComponents}`);

    const split = tf.splitIntoComponents(combined, cc.labels);
    assert(split.components.length === 2, `expected 2 components, got ${split.components.length}`);
    for (const c of split.components)
      assert(c.dtype === "float64", "component dtype is float64");

    const totalFaces = split.components[0].numberOfFaces + split.components[1].numberOfFaces;
    assert(totalFaces === combined.numberOfFaces,
      `component faces ${totalFaces} != combined ${combined.numberOfFaces}`);
    log(`  splitIntoComponents (f64): ${split.components.length} components`, "line-pass");

    for (const c of split.components) c.delete();
    split.labels.delete();
    cc.labels.delete();
    combined.delete();
    s2.delete();
    s1.delete();
  });

  // ==========================================================================
  test("async: reindexedByMask (float64)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });
    const nf = sphere.numberOfFaces;

    const maskData = new Int8Array(nf);
    for (let i = 0; i < Math.floor(nf / 2); i++) maskData[i] = 1;
    const mask = tf.ndarray(maskData, [nf]);

    const filtered = await tf.async.reindexedByMask(sphere, mask);
    assert(filtered.dtype === "float64", "result dtype is float64");
    assert(filtered.numberOfFaces === Math.floor(nf / 2),
      `expected ${Math.floor(nf / 2)} faces, got ${filtered.numberOfFaces}`);
    log(`  async reindexedByMask (f64): ${filtered.numberOfFaces} faces`, "line-pass");

    filtered.delete();
    mask.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("async: concatenateMeshes (float64)", async () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 8, 8, { dtype: "float64" });
    const s2 = tf.sphereMesh(0.5, 6, 6, { dtype: "float64" });

    const combined = await tf.async.concatenateMeshes(s1, s2);
    assert(combined.dtype === "float64", "result dtype is float64");
    assert(combined.numberOfFaces === s1.numberOfFaces + s2.numberOfFaces,
      `expected ${s1.numberOfFaces + s2.numberOfFaces} faces, got ${combined.numberOfFaces}`);
    log(`  async concatenateMeshes (f64): ${combined.numberOfFaces} faces`, "line-pass");

    combined.delete();
    s2.delete();
    s1.delete();
  });

  // ==========================================================================
  test("async: splitIntoComponents (float64)", async () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 8, 8, { dtype: "float64" });
    const s2 = tf.sphereMesh(0.5, 6, 6, { dtype: "float64" });
    const combined = tf.concatenateMeshes(s1, s2);

    const cc = tf.connectedComponents(combined, "edge");
    const split = await tf.async.splitIntoComponents(combined, cc.labels);
    assert(split.components.length === 2, `expected 2 components, got ${split.components.length}`);
    for (const c of split.components)
      assert(c.dtype === "float64", "component dtype is float64");
    log(`  async splitIntoComponents (f64): ${split.components.length} components`, "line-pass");

    for (const c of split.components) c.delete();
    split.labels.delete();
    cc.labels.delete();
    combined.delete();
    s2.delete();
    s1.delete();
  });

});

describe("Reindex (dtype mismatch)", () => {

  test("concatenateMeshes dtype mismatch throws", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { tf.concatenateMeshes(s0, s1); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  concatenateMeshes dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

  test("async: concatenateMeshes dtype mismatch throws", async () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { await tf.async.concatenateMeshes(s0, s1); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  async concatenateMeshes dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

});

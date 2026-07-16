import { describe, test, log, assert, getTf } from "./harness.mjs";

/** Helper: serialize mesh to Uint8Array via writeStl/writeObj. */
function toBytes(tf, mesh, writeFn) {
  const raw = writeFn(mesh);
  const buf = new Uint8Array(raw.data.length);
  for (let i = 0; i < buf.length; i++) buf[i] = raw.data[i];
  raw.delete();
  return buf;
}

describe("I/O write + roundtrip", () => {

  test("writeStl returns bytes", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const stlBytes = tf.writeStl(mesh);

    assert(stlBytes.length > 0, `STL bytes: ${stlBytes.length}`);
    assert(stlBytes.dtype === "int8", `dtype: ${stlBytes.dtype}`);
    log(`  writeStl: ${stlBytes.length} bytes`, "line-pass");

    stlBytes.delete(); mesh.delete();
  });

  test("STL roundtrip", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const originalFaces = mesh.numberOfFaces;

    const stlBytes = tf.writeStl(mesh);
    // readStl expects ArrayBuffer or Uint8Array
    const buf = new Uint8Array(stlBytes.data.length);
    for (let i = 0; i < buf.length; i++) buf[i] = stlBytes.data[i];
    const m2 = tf.readStl(buf.buffer);

    assert(m2.numberOfFaces === originalFaces,
      `faces: ${m2.numberOfFaces} === ${originalFaces}`);
    log(`  STL roundtrip: ${originalFaces} faces preserved`, "line-pass");

    m2.delete(); stlBytes.delete(); mesh.delete();
  });

  test("writeObj returns bytes", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const objBytes = tf.writeObj(mesh);

    assert(objBytes.length > 0, `OBJ bytes: ${objBytes.length}`);
    assert(objBytes.dtype === "int8", `dtype: ${objBytes.dtype}`);
    log(`  writeObj: ${objBytes.length} bytes`, "line-pass");

    objBytes.delete(); mesh.delete();
  });

  test("OBJ roundtrip", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const originalFaces = mesh.numberOfFaces;

    const objBytes = tf.writeObj(mesh);
    const buf = new Uint8Array(objBytes.data.length);
    for (let i = 0; i < buf.length; i++) buf[i] = objBytes.data[i];
    const m2 = tf.readObj(buf.buffer);

    assert(m2.numberOfFaces === originalFaces,
      `faces: ${m2.numberOfFaces} === ${originalFaces}`);
    log(`  OBJ roundtrip: ${originalFaces} faces preserved`, "line-pass");

    m2.delete(); objBytes.delete(); mesh.delete();
  });

  // ==========================================================================
  test("async: STL roundtrip", async () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const originalFaces = mesh.numberOfFaces;

    const stlBytes = await tf.async.writeStl(mesh);
    const buf = new Uint8Array(stlBytes.data.length);
    for (let i = 0; i < buf.length; i++) buf[i] = stlBytes.data[i];
    const m2 = await tf.async.readStl(buf.buffer);

    assert(m2.numberOfFaces === originalFaces,
      `faces: ${m2.numberOfFaces} === ${originalFaces}`);
    log(`  async STL roundtrip: ${originalFaces} faces preserved`, "line-pass");

    m2.delete(); stlBytes.delete(); mesh.delete();
  });

  // ==========================================================================
  test("async: OBJ roundtrip", async () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const originalFaces = mesh.numberOfFaces;

    const objBytes = await tf.async.writeObj(mesh);
    const buf = new Uint8Array(objBytes.data.length);
    for (let i = 0; i < buf.length; i++) buf[i] = objBytes.data[i];
    const m2 = await tf.async.readObj(buf.buffer);

    assert(m2.numberOfFaces === originalFaces,
      `faces: ${m2.numberOfFaces} === ${originalFaces}`);
    log(`  async OBJ roundtrip: ${originalFaces} faces preserved`, "line-pass");

    m2.delete(); objBytes.delete(); mesh.delete();
  });

});

describe("readStlData / readObjData", () => {

  test("readStlData returns MeshLike with correct shapes", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const nf = mesh.numberOfFaces;
    const np = mesh.numberOfPoints;
    const buf = toBytes(tf, mesh, (m) => tf.writeStl(m));

    const data = tf.readStlData(buf);

    assert(data.faces != null, "faces is defined");
    assert(data.points != null, "points is defined");
    assert(data.faces.dtype === "int32", `faces dtype: ${data.faces.dtype}`);
    assert(data.points.dtype === "float32", `points dtype: ${data.points.dtype}`);
    assert(data.faces.shape[0] === nf, `faces rows: ${data.faces.shape[0]} === ${nf}`);
    assert(data.faces.shape[1] === 3, `faces cols: ${data.faces.shape[1]} === 3`);
    assert(data.points.shape[0] === np, `points rows: ${data.points.shape[0]} === ${np}`);
    assert(data.points.shape[1] === 3, `points cols: ${data.points.shape[1]} === 3`);
    log(`  readStlData: ${nf} faces, ${np} points`, "line-pass");

    data.faces.delete(); data.points.delete(); mesh.delete();
  });

  test("readObjData returns MeshLike with NDArrayInt32 faces", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const nf = mesh.numberOfFaces;
    const np = mesh.numberOfPoints;
    const buf = toBytes(tf, mesh, (m) => tf.writeObj(m));

    const data = tf.readObjData(buf);

    assert(data.faces.dtype === "int32", `faces dtype: ${data.faces.dtype}`);
    assert(data.points.dtype === "float32", `points dtype: ${data.points.dtype}`);
    assert(data.faces.shape[0] === nf, `faces rows: ${data.faces.shape[0]} === ${nf}`);
    assert(data.faces.shape[1] === 3, `faces cols: ${data.faces.shape[1]} === 3`);
    assert(data.points.shape[0] === np, `points rows: ${data.points.shape[0]} === ${np}`);
    log(`  readObjData: ${nf} faces, ${np} points`, "line-pass");

    data.faces.delete(); data.points.delete(); mesh.delete();
  });

  test("readObjData({ dynamic }) returns OffsetBlockedBuffer faces", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const nf = mesh.numberOfFaces;
    const buf = toBytes(tf, mesh, (m) => tf.writeObj(m));

    const data = tf.readObjData(buf, { dynamic: true });

    // OffsetBlockedBuffer has .length and .offsets, not .shape
    assert(data.faces.length === nf, `faces blocks: ${data.faces.length} === ${nf}`);
    assert(data.faces.offsets != null, "faces has offsets (OffsetBlockedBuffer)");
    assert(data.points.dtype === "float32", `points dtype: ${data.points.dtype}`);
    log(`  readObjData dynamic: ${nf} polygons`, "line-pass");

    data.faces.delete(); data.points.delete(); mesh.delete();
  });

  test("readObj({ dynamic }) triangulates correctly", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const nf = mesh.numberOfFaces;
    const buf = toBytes(tf, mesh, (m) => tf.writeObj(m));

    // OBJ from writeObj is all triangles, so dynamic read + triangulate
    // should produce the same face count
    const m2 = tf.readObj(buf, { dynamic: true });
    assert(m2.numberOfFaces === nf,
      `dynamic roundtrip faces: ${m2.numberOfFaces} === ${nf}`);
    log(`  readObj dynamic roundtrip: ${nf} faces`, "line-pass");

    m2.delete(); mesh.delete();
  });

  test("readStlData → triangulate → Mesh roundtrip", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const nf = mesh.numberOfFaces;
    const buf = toBytes(tf, mesh, (m) => tf.writeStl(m));

    const data = tf.readStlData(buf);
    const m2 = tf.triangulate(data);

    assert(m2.numberOfFaces === nf,
      `triangulated faces: ${m2.numberOfFaces} === ${nf}`);
    log(`  readStlData → triangulate: ${nf} faces`, "line-pass");

    data.faces.delete(); data.points.delete(); m2.delete(); mesh.delete();
  });

  test("readObjData({ dynamic }) → triangulate roundtrip", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const nf = mesh.numberOfFaces;
    const buf = toBytes(tf, mesh, (m) => tf.writeObj(m));

    const data = tf.readObjData(buf, { dynamic: true });
    const m2 = tf.triangulate(data);

    assert(m2.numberOfFaces === nf,
      `triangulated faces: ${m2.numberOfFaces} === ${nf}`);
    log(`  readObjData dynamic → triangulate: ${nf} faces`, "line-pass");

    data.faces.delete(); data.points.delete(); m2.delete(); mesh.delete();
  });

});

describe("I/O float64", () => {

  test("readObj({ dtype: 'float64' }) produces float64 mesh", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const buf = toBytes(tf, mesh, (m) => tf.writeObj(m));

    const m2 = tf.readObj(buf, { dtype: "float64" });
    assert(m2.dtype === "float64", `read mesh dtype: ${m2.dtype}`);
    assert(m2.points.dtype === "float64", `points dtype: ${m2.points.dtype}`);
    assert(m2.numberOfFaces === mesh.numberOfFaces,
      `faces: ${m2.numberOfFaces} === ${mesh.numberOfFaces}`);
    log(`  float64 OBJ roundtrip: ${mesh.numberOfFaces} faces`, "line-pass");

    m2.delete(); mesh.delete();
  });

  test("readObj() defaults to float32 when no dtype given", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const buf = toBytes(tf, mesh, (m) => tf.writeObj(m));

    const m2 = tf.readObj(buf);
    assert(m2.dtype === "float32", `default dtype: ${m2.dtype}`);
    log(`  default OBJ read dtype = float32`, "line-pass");

    m2.delete(); mesh.delete();
  });

  test("writeObj(float64) emits %.17g precision", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const objBytes = tf.writeObj(mesh);

    // Convert to string and look for a long decimal — float32 %.9g would top
    // out around 9 significant digits, float64 %.17g goes to 17.
    const buf = new Uint8Array(objBytes.data.length);
    for (let i = 0; i < buf.length; i++) buf[i] = objBytes.data[i];
    const text = new TextDecoder().decode(buf);
    // Find at least one numeric token with > 9 significant digits.
    const longFloat = /\b\d\.\d{10,}\b/.test(text);
    assert(longFloat, `expected high-precision floats in OBJ output`);
    log(`  float64 OBJ emits long decimals`, "line-pass");

    objBytes.delete(); mesh.delete();
  });

  test("writeStl accepts float64 mesh (STL output is float32 by spec)", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const stlBytes = tf.writeStl(mesh);

    assert(stlBytes.length > 0, `STL bytes: ${stlBytes.length}`);
    // Roundtrip: STL read always yields float32
    const buf = new Uint8Array(stlBytes.data.length);
    for (let i = 0; i < buf.length; i++) buf[i] = stlBytes.data[i];
    const m2 = tf.readStl(buf.buffer);
    assert(m2.dtype === "float32", `STL read dtype: ${m2.dtype}`);
    assert(m2.numberOfFaces === mesh.numberOfFaces,
      `faces: ${m2.numberOfFaces} === ${mesh.numberOfFaces}`);
    log(`  float64 → STL → float32 mesh: ${mesh.numberOfFaces} faces`, "line-pass");

    m2.delete(); stlBytes.delete(); mesh.delete();
  });

  test("readObjData({ dtype: 'float64' }) returns float64 points", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const buf = toBytes(tf, mesh, (m) => tf.writeObj(m));

    const data = tf.readObjData(buf, { dtype: "float64" });
    assert(data.points.dtype === "float64", `points dtype: ${data.points.dtype}`);
    assert(data.faces.dtype === "int32", `faces dtype: ${data.faces.dtype}`);
    log(`  readObjData float64: points.dtype=${data.points.dtype}`, "line-pass");

    data.faces.delete(); data.points.delete(); mesh.delete();
  });

  test("async: readObj({ dtype: 'float64' }) roundtrip", async () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const buf = toBytes(tf, mesh, (m) => tf.writeObj(m));

    const m2 = await tf.async.readObj(buf, { dtype: "float64" });
    assert(m2.dtype === "float64", `async read dtype: ${m2.dtype}`);
    assert(m2.numberOfFaces === mesh.numberOfFaces,
      `faces: ${m2.numberOfFaces} === ${mesh.numberOfFaces}`);
    log(`  async float64 OBJ roundtrip: ${mesh.numberOfFaces} faces`, "line-pass");

    m2.delete(); mesh.delete();
  });

  test("async: writeObj(float64) returns bytes", async () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const objBytes = await tf.async.writeObj(mesh);

    assert(objBytes.length > 0, `OBJ bytes: ${objBytes.length}`);
    log(`  async writeObj float64: ${objBytes.length} bytes`, "line-pass");

    objBytes.delete(); mesh.delete();
  });

  test("async: writeStl(float64) returns bytes", async () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const stlBytes = await tf.async.writeStl(mesh);

    assert(stlBytes.length > 0, `STL bytes: ${stlBytes.length}`);
    log(`  async writeStl float64: ${stlBytes.length} bytes`, "line-pass");

    stlBytes.delete(); mesh.delete();
  });

  test("writes apply the mesh transformation (sync == async)", async () => {
    const tf = getTf();
    for (const dtype of ["float32", "float64"]) {
      const mesh = tf.sphereMesh(1, 8, 8, { dtype });
      mesh.transformation = tf.makeTranslation(10, 0, 0);

      const syncStl = tf.writeStl(mesh);
      const asyncStl = await tf.async.writeStl(mesh);
      assert(syncStl.length === asyncStl.length,
             `stl length sync ${syncStl.length} == async ${asyncStl.length}`);
      const a = syncStl.data, b = asyncStl.data;
      let same = true;
      for (let i = 0; i < a.length; ++i)
        if (a[i] !== b[i]) { same = false; break; }
      assert(same, `stl bytes identical (${dtype})`);

      // the translated sphere's re-read x-range must sit around 10
      const rm = tf.readStl(syncStl.data);
      const pts = rm.points.data;
      let minX = Infinity;
      for (let i = 0; i < pts.length; i += 3) minX = Math.min(minX, pts[i]);
      assert(minX > 8, `matrix applied: min x ${minX} > 8 (${dtype})`);
      rm.delete();

      const syncObj = tf.writeObj(mesh);
      const asyncObj = await tf.async.writeObj(mesh);
      let sameObj = syncObj.length === asyncObj.length;
      if (sameObj) {
        const c = syncObj.data, d = asyncObj.data;
        for (let i = 0; i < c.length; ++i)
          if (c[i] !== d[i]) { sameObj = false; break; }
      }
      assert(sameObj, `obj bytes identical (${dtype})`);

      log(`  transformed write sync==async (${dtype})`, "line-pass");
      syncStl.delete(); asyncStl.delete(); syncObj.delete(); asyncObj.delete();
      mesh.delete();
    }
  });

});

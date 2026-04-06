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

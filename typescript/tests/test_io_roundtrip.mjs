import { describe, test, log, assert, getTf } from "./harness.mjs";

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

});

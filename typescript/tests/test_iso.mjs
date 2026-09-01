import { describe, test, log, assert, getTf } from "./harness.mjs";

describe("Isocontours", () => {

  test("isocontours single threshold", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0); // x-coords as scalar field

    const curves = tf.isocontours(plane, scalars, 0.0);
    assert(curves.length > 0, `curves count: ${curves.length}`);
    assert(curves.points.shape[1] === 3, "3D points");
    log(`  isocontours(single): ${curves.length} curves`, "line-pass");

    curves.delete(); scalars.delete(); plane.delete();
  });

  test("isocontours multiple thresholds", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const thresholds = new Float32Array([-1, 0, 1]);

    const curves = tf.isocontours(plane, scalars, thresholds);
    assert(curves.length > 0, `curves count: ${curves.length}`);
    log(`  isocontours(multi): ${curves.length} curves`, "line-pass");

    curves.delete(); scalars.delete(); plane.delete();
  });

  test("async: isocontours", async () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);

    const curves = await tf.async.isocontours(plane, scalars, 0.0);
    assert(curves.length > 0, `curves count: ${curves.length}`);
    assert(curves.points.shape[1] === 3, "3D points");
    log(`  async isocontours: ${curves.length} curves`, "line-pass");

    curves.delete(); scalars.delete(); plane.delete();
  });

});

describe("Isobands", () => {

  test("isobands basic", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0); // x-coords as scalar field
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues);
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  isobands: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with curves", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, { returnCurves: true });
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  isobands with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with selectedBands", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, { selectedBands: [1] });
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  isobands selectedBands: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with selectedBands + curves", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, {
      selectedBands: [1],
      returnCurves: true,
    });
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    log("  isobands selectedBands + curves", "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("async: isobands", async () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = await tf.async.isobands(plane, scalars, cutValues);
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  async isobands: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

});

describe("Isocontours (float64)", () => {

  test("isocontours single threshold (float64)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20, { dtype: "float64" });
    const scalars = plane.points.take(null, 0);

    const curves = tf.isocontours(plane, scalars, 0.0);
    assert(curves.dtype === "float64", "result curves dtype is float64");
    assert(curves.length > 0, `curves count: ${curves.length}`);
    assert(curves.points.shape[1] === 3, "3D points");
    log(`  isocontours(single, f64): ${curves.length} curves`, "line-pass");

    curves.delete(); scalars.delete(); plane.delete();
  });

  test("isocontours multiple thresholds (float64)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20, { dtype: "float64" });
    const scalars = plane.points.take(null, 0);
    const thresholds = new Float64Array([-1, 0, 1]);

    const curves = tf.isocontours(plane, scalars, thresholds);
    assert(curves.dtype === "float64", "result curves dtype is float64");
    assert(curves.length > 0, `curves count: ${curves.length}`);
    log(`  isocontours(multi, f64): ${curves.length} curves`, "line-pass");

    curves.delete(); scalars.delete(); plane.delete();
  });

  test("async: isocontours (float64)", async () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20, { dtype: "float64" });
    const scalars = plane.points.take(null, 0);

    const curves = await tf.async.isocontours(plane, scalars, 0.0);
    assert(curves.dtype === "float64", "result curves dtype is float64");
    assert(curves.length > 0, `curves count: ${curves.length}`);
    assert(curves.points.shape[1] === 3, "3D points");
    log(`  async isocontours (f64): ${curves.length} curves`, "line-pass");

    curves.delete(); scalars.delete(); plane.delete();
  });

});

describe("Isobands (float64)", () => {

  test("isobands basic (float64)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20, { dtype: "float64" });
    const scalars = plane.points.take(null, 0); // x-coords as scalar field
    const cutValues = new Float64Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues);
    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  isobands (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with curves (float64)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20, { dtype: "float64" });
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float64Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, { returnCurves: true });
    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.curves.dtype === "float64", "result curves dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  isobands with curves (f64): ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with selectedBands (float64)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20, { dtype: "float64" });
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float64Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, { selectedBands: [1] });
    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  isobands selectedBands (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with selectedBands + curves (float64)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20, { dtype: "float64" });
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float64Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, {
      selectedBands: [1],
      returnCurves: true,
    });
    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.curves.dtype === "float64", "result curves dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    log("  isobands selectedBands + curves (f64)", "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("async: isobands (float64)", async () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20, { dtype: "float64" });
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float64Array([-1, 0, 1]);

    const result = await tf.async.isobands(plane, scalars, cutValues);
    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  async isobands (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

});

describe("Isobands (dtype mismatch)", () => {

  test("isobands dtype mismatch throws", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars64 = tf.zeros("float64", [plane.points.shape[0]]);
    const cutValues = new Float32Array([-1, 0, 1]);
    let threw = false;
    try { tf.isobands(plane, scalars64, cutValues); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  isobands dtype mismatch throws`, "line-pass");
    scalars64.delete(); plane.delete();
  });

  test("async: isobands dtype mismatch throws", async () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars64 = tf.zeros("float64", [plane.points.shape[0]]);
    const cutValues = new Float32Array([-1, 0, 1]);
    let threw = false;
    try { await tf.async.isobands(plane, scalars64, cutValues); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  async isobands dtype mismatch throws`, "line-pass");
    scalars64.delete(); plane.delete();
  });

});

describe("Isocontours (dtype mismatch)", () => {

  test("isocontours dtype mismatch throws", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars64 = tf.zeros("float64", [plane.points.shape[0]]);
    let threw = false;
    try { tf.isocontours(plane, scalars64, 0.0); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  isocontours dtype mismatch throws`, "line-pass");
    scalars64.delete(); plane.delete();
  });

  test("async: isocontours dtype mismatch throws", async () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars64 = tf.zeros("float64", [plane.points.shape[0]]);
    let threw = false;
    try { await tf.async.isocontours(plane, scalars64, 0.0); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  async isocontours dtype mismatch throws`, "line-pass");
    scalars64.delete(); plane.delete();
  });

});

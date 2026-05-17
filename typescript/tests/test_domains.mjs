import { describe, test, log, assert, getTf } from "./harness.mjs";

const F64 = { dtype: "float64" };
const TOL = 1e-8;

// ---- helpers ---------------------------------------------------------------

function translate(mesh, dx, dy, dz) {
  const tf = getTf();
  mesh.points.add_(tf.ndarray([dx, dy, dz], [3]));
}

/**
 * Apply a 3x3 row-major transformation matrix to every point at once.
 * `m9` is the flat 9-element matrix; result = points @ M.
 */
function applyMatrix(mesh, m9) {
  const tf = getTf();
  const M = tf.ndarray(m9, [3, 3]);
  mesh.points = mesh.points.matMul(M);
}

// Permutation matrices for plane reorientation (XY plane → XZ or YZ plane).
// Row-major.  output_col_i = sum_j input_col_j * M[j][i]
//  XY → XZ (swap y, z; +y normal):
const M_TO_XZ = [1, 0, 0,  0, 0, 1,  0, 1, 0];
//  XY → YZ (swap x, z; +x normal):
const M_TO_YZ = [0, 0, 1,  0, 1, 0,  1, 0, 0];

/** Run full pipeline → sorted per-domain signed volumes. */
function sortedDomainVolumes(meshes) {
  const tf = getTf();
  const arr = tf.meshArrangements(meshes);
  const cleaned = tf.cleaned(arr.mesh, 1e-6);
  const dl = tf.domainLabels(cleaned, { ignoreOpenFragments: true });
  const split = tf.splitIntoDomains(cleaned, dl);
  const vols = split.components.map((sub) => tf.signedVolume(sub));
  vols.sort((a, b) => a - b);

  // cleanup
  for (const sub of split.components) sub.delete();
  split.labels.delete();
  dl.labels.delete();
  cleaned.delete();
  arr.faceLabels.delete();
  arr.tagLabels.delete();
  arr.mesh.delete();
  return vols;
}

function assertVolumesMatch(got, expected, name) {
  assert(got.length === expected.length,
    `${name}: expected ${expected.length} domains, got ${got.length}: [${got.join(", ")}]`);
  for (let i = 0; i < got.length; i++) {
    const diff = Math.abs(got[i] - expected[i]);
    assert(diff < TOL,
      `${name}: vol[${i}] = ${got[i]}, expected ${expected[i]} (diff ${diff})`);
  }
  log(`  ${name}: ${got.length} domains ✓`, "line-pass");
}

// ---- tests -----------------------------------------------------------------

describe("Domain extraction (cross-runtime correctness)", () => {

  test("case 1: sphere + plane", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 32, 32, F64);
    const plane = tf.planeMesh(3.0, 3.0, 1, 1, F64);
    const got = sortedDomainVolumes([sphere, plane]);
    sphere.delete(); plane.delete();
    assertVolumesMatch(got, [
      -4.15190646195,
       2.07595323098,
       2.07595323098,
    ], "case 1");
  });

  test("case 2: sphere + 2 planes z=±0.3", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 32, 32, F64);
    const top = tf.planeMesh(3.0, 3.0, 1, 1, F64);   translate(top, 0, 0,  0.3);
    const bot = tf.planeMesh(3.0, 3.0, 1, 1, F64);   translate(bot, 0, 0, -0.3);
    const got = sortedDomainVolumes([sphere, top, bot]);
    sphere.delete(); top.delete(); bot.delete();
    assertVolumesMatch(got, [
      -4.15190646195,
       1.16908135492,
       1.16908135492,
       1.81374375211,
    ], "case 2");
  });

  test("case 3: two disjoint sphere+plane copies", () => {
    const tf = getTf();
    const s1 = tf.sphereMesh(1.0, 32, 32, F64);
    const p1 = tf.planeMesh(3.0, 3.0, 1, 1, F64);
    const s2 = tf.sphereMesh(1.0, 32, 32, F64);  translate(s2, 3, 0, 0);
    const p2 = tf.planeMesh(3.0, 3.0, 1, 1, F64); translate(p2, 3, 0, 0);
    const got = sortedDomainVolumes([s1, p1, s2, p2]);
    s1.delete(); p1.delete(); s2.delete(); p2.delete();
    assertVolumesMatch(got, [
      -8.30381292391,
       2.07595323098,
       2.07595323098,
       2.07595323098,
       2.07595323098,
    ], "case 3");
  });

  test("case 4: sphere + 2 horiz + 2 vert planes", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 32, 32, F64);
    const top = tf.planeMesh(3.0, 3.0, 1, 1, F64);   translate(top, 0, 0,  0.3);
    const bot = tf.planeMesh(3.0, 3.0, 1, 1, F64);   translate(bot, 0, 0, -0.3);
    const xz = tf.planeMesh(3.0, 3.0, 1, 1, F64);    applyMatrix(xz, M_TO_XZ);
    const yz = tf.planeMesh(3.0, 3.0, 1, 1, F64);    applyMatrix(yz, M_TO_YZ);
    const got = sortedDomainVolumes([sphere, top, bot, xz, yz]);
    sphere.delete(); top.delete(); bot.delete(); xz.delete(); yz.delete();
    assertVolumesMatch(got, [
      -4.15190646195,
       0.292270338731, 0.292270338731, 0.292270338731, 0.292270338731,
       0.292270338731, 0.292270338731, 0.292270338731, 0.292270338731,
       0.453435938026, 0.453435938026, 0.453435938026, 0.453435938026,
    ], "case 4");
  });

  test("case 6: big sphere + plane + small sphere at origin", () => {
    const tf = getTf();
    const big = tf.sphereMesh(1.0, 32, 32, F64);
    const plane = tf.planeMesh(3.0, 3.0, 1, 1, F64);
    const small = tf.sphereMesh(0.2, 16, 16, F64);
    const got = sortedDomainVolumes([big, plane, small]);
    big.delete(); plane.delete(); small.delete();
    assertVolumesMatch(got, [
      -4.15190646195,
       0.0161709591442,
       0.0161709591442,
       2.05978227183,
       2.05978227183,
    ], "case 6");
  });

  test("case 7: big sphere + plane + small sphere nested at z=-0.5", () => {
    const tf = getTf();
    const big = tf.sphereMesh(1.0, 32, 32, F64);
    const plane = tf.planeMesh(3.0, 3.0, 1, 1, F64);
    const small = tf.sphereMesh(0.2, 16, 16, F64);   translate(small, 0, 0, -0.5);
    const got = sortedDomainVolumes([big, plane, small]);
    big.delete(); plane.delete(); small.delete();
    assertVolumesMatch(got, [
      -4.15190646195,
       0.0323419182883,
       2.04361131269,
       2.07595323098,
    ], "case 7");
  });

});

describe("Domain extraction (async dispatch)", () => {

  test("async: sphere + plane (same as case 1)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 32, 32, F64);
    const plane = tf.planeMesh(3.0, 3.0, 1, 1, F64);

    const arr = await tf.async.meshArrangements([sphere, plane]);
    const cleaned = await tf.async.cleaned(arr.mesh, 1e-6);
    const dl = await tf.async.domainLabels(cleaned, { ignoreOpenFragments: true });
    const split = await tf.async.splitIntoDomains(cleaned, dl);
    const vols = split.components.map((sub) => tf.signedVolume(sub));
    vols.sort((a, b) => a - b);

    sphere.delete(); plane.delete();
    for (const sub of split.components) sub.delete();
    split.labels.delete();
    dl.labels.delete();
    cleaned.delete();
    arr.faceLabels.delete();
    arr.tagLabels.delete();
    arr.mesh.delete();

    assertVolumesMatch(vols, [
      -4.15190646195,
       2.07595323098,
       2.07595323098,
    ], "async case 1");
  });

});

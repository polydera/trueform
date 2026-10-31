import { describe, test, log, assert, getTf } from "./harness.mjs";

function throws(fn, Ctor, msg) {
  try {
    fn();
  } catch (e) {
    assert(e instanceof Ctor, `${msg}: expected ${Ctor.name}, got ${e}`);
    return;
  }
  assert(false, `${msg}: did not throw`);
}

function findPoint(points, x, y) {
  for (let k = 0; k < points.length / 3; k++)
    if (points[3 * k] === x && points[3 * k + 1] === y) return k;
  return -1;
}

function ids(tf, list) {
  return tf.ndarray(Int32Array.from(list), [list.length]);
}

function values(tf, list) {
  return tf.ndarray(Float64Array.from(list), [list.length]);
}

/** The points within `rings` grid steps of a 4 x 4 plane's rim. */
function rimHeld(points, linear, rings, ticks) {
  const edge = 2.0 - (rings - 1) * (4.0 / ticks) - 1e-6;
  const held = [], vals = [];
  for (let k = 0; k < points.length / 3; k++) {
    const x = points[3 * k], y = points[3 * k + 1];
    if (Math.abs(x) >= edge || Math.abs(y) >= edge) {
      held.push(k);
      vals.push(linear(x, y));
    }
  }
  return [held, vals];
}

/** A deterministic jittered cap with stated positive weights. */
function jitteredCap(tf) {
  const plane = tf.planeMesh(1.0, 1.0, 9, 9, { dtype: "float64" });
  const faces = Int32Array.from(plane.faces.data.slice());
  const points = Float64Array.from(plane.points.data.slice());
  plane.delete();
  const n = points.length / 3;
  const held = [], vals = [];
  for (let k = 0; k < n; k++) {
    const x = points[3 * k], y = points[3 * k + 1];
    const inside = Math.abs(x) < 0.49 && Math.abs(y) < 0.49;
    if (inside) {
      points[3 * k] += 0.02 * Math.sin(12.9898 * k);
      points[3 * k + 1] += 0.02 * Math.cos(78.233 * k);
    } else {
      held.push(k);
      vals.push((x > 0 ? 1.0 : 0.0) + y);
    }
    const px = points[3 * k], py = points[3 * k + 1];
    points[3 * k + 2] = 0.3 * (1.0 - 2.0 * (px * px + py * py));
  }
  const weights = new Float64Array(faces.length);
  for (let f = 0; f < faces.length / 3; f++)
    for (let c = 0; c < 3; c++) weights[3 * f + c] = 0.5 + ((7 * f + 3 * c) % 11) / 4;
  return { faces, points, weights, held, vals };
}

/** The same system solved densely: L from the stated weights, M lumped. */
function denseField(cap, order) {
  const { faces, points, weights, held, vals } = cap;
  const n = points.length / 3;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  const mass = new Float64Array(n);
  for (let f = 0; f < faces.length / 3; f++) {
    for (let c = 0; c < 3; c++) {
      const i = faces[3 * f + (c + 1) % 3], j = faces[3 * f + (c + 2) % 3];
      const w = weights[3 * f + c];
      L[i][j] -= w; L[j][i] -= w; L[i][i] += w; L[j][j] += w;
    }
    const [a, b, d] = [faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]];
    const u = [0, 1, 2].map((t) => points[3 * b + t] - points[3 * a + t]);
    const v = [0, 1, 2].map((t) => points[3 * d + t] - points[3 * a + t]);
    const cx = u[1] * v[2] - u[2] * v[1], cy = u[2] * v[0] - u[0] * v[2], cz = u[0] * v[1] - u[1] * v[0];
    const area = 0.5 * Math.hypot(cx, cy, cz);
    for (const k of [a, b, d]) mass[k] += area / 3;
  }
  let A = L;
  if (order === 2) {
    A = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++)
      for (let k = 0; k < n; k++) {
        const lik = L[i][k] / mass[k];
        if (lik === 0) continue;
        for (let j = 0; j < n; j++) A[i][j] += lik * L[k][j];
      }
  }
  const isHeld = new Uint8Array(n);
  const x = new Float64Array(n);
  held.forEach((k, t) => { isHeld[k] = 1; x[k] = vals[t]; });
  const free = [];
  for (let k = 0; k < n; k++) if (!isHeld[k]) free.push(k);
  const m = free.length;
  const M = free.map((i) => {
    const row = new Float64Array(m + 1);
    free.forEach((j, c) => { row[c] = A[i][j]; });
    let rhs = 0;
    held.forEach((k) => { rhs -= A[i][k] * x[k]; });
    row[m] = rhs;
    return row;
  });
  for (let c = 0; c < m; c++) {
    let p = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < m; r++) {
      const f = M[r][c] / M[c][c];
      if (f === 0) continue;
      for (let k = c; k <= m; k++) M[r][k] -= f * M[c][k];
    }
  }
  for (let c = m - 1; c >= 0; c--) {
    let s = M[c][m];
    for (let k = c + 1; k < m; k++) s -= M[c][k] * x[free[k]];
    x[free[c]] = s / M[c][c];
  }
  return x;
}

describe("Surface fields", () => {

  for (const dtype of ["float32", "float64"]) {
    test(`geodesicDistances on a plane is the straight line (${dtype})`, () => {
      const tf = getTf();
      const mesh = tf.planeMesh(4.0, 4.0, 8, 8, { dtype });
      const points = mesh.points.data.slice();
      const seed = findPoint(points, -2.0, -2.0);
      assert(seed >= 0, "corner seed");
      const seeds = ids(tf, [seed]);
      const d = tf.geodesicDistances(mesh, seeds);
      assert(d.dtype === dtype, "distances in the mesh's dtype");
      assert(d.length === mesh.numberOfPoints, "one distance per point");
      const dd = d.data.slice();
      for (let k = 0; k < mesh.numberOfPoints; k++) {
        const line = Math.hypot(points[3 * k] - points[3 * seed], points[3 * k + 1] - points[3 * seed + 1]);
        assert(Math.abs(dd[k] - line) <= 1e-6 * (1 + line), `point ${k}: ${dd[k]} vs ${line}`);
      }
      assert(dd[seed] === 0, "zero at the seed");
      d.delete(); seeds.delete(); mesh.delete();
    });

    test(`geodesicDistances on a sphere answer the arc (${dtype})`, () => {
      const tf = getTf();
      const radius = 2.0;
      const mesh = tf.sphereMesh(radius, 60, 80, { dtype });
      const p = mesh.points.data.slice();
      let seed = 0;
      for (let k = 0; k < p.length / 3; k++) if (Math.abs(p[3 * k + 2]) < Math.abs(p[3 * seed + 2])) seed = k;
      const seeds = ids(tf, [seed]);
      const d = tf.geodesicDistances(mesh, seeds).data.slice();
      let worst = 0;
      for (let k = 0; k < p.length / 3; k++) {
        const inner = p[3 * k] * p[3 * seed] + p[3 * k + 1] * p[3 * seed + 1] + p[3 * k + 2] * p[3 * seed + 2];
        const arc = radius * Math.acos(Math.min(1, Math.max(-1, inner / (radius * radius))));
        if (arc < 0.5 || arc > 2.5 * radius) continue;
        worst = Math.max(worst, Math.abs(d[k] - arc) / arc);
      }
      assert(worst < 0.01, `worst relative error ${worst}`);
      log(`  ${dtype}: worst arc error ${worst.toExponential(2)}`, "line-pass");
      seeds.delete(); mesh.delete();
    });
  }

  test("geodesicDistances answer the nearest seed, a reach, and no seed", () => {
    const tf = getTf();
    const mesh = tf.planeMesh(4.0, 4.0, 16, 16, { dtype: "float64" });
    const points = mesh.points.data.slice();
    const n = points.length / 3;
    const dist = (k, s) => Math.hypot(points[3 * k] - points[3 * s], points[3 * k + 1] - points[3 * s + 1]);
    const a = findPoint(points, -2.0, -2.0), b = findPoint(points, 2.0, 2.0), c = findPoint(points, 0.0, 0.0);
    const both = ids(tf, [a, b]), centre = ids(tf, [c]), none = ids(tf, []);
    const d = tf.geodesicDistances(mesh, both).data.slice();
    let apart = 0;
    for (let k = 0; k < n; k++) {
      const first = dist(k, a), second = dist(k, b), nearest = Math.min(first, second);
      // Where the two fronts meet the answer stays near, not exact.
      if (nearest < 0.85 * Math.max(first, second)) {
        assert(Math.abs(d[k] - nearest) <= 1e-9 * (1 + nearest), `nearest at ${k}`);
        ++apart;
      } else assert(Math.abs(d[k] - nearest) <= 0.03 * nearest, `meeting at ${k}`);
    }
    assert(apart > 200, `apart ${apart}`);
    const reached = tf.geodesicDistances(mesh, centre, { radius: 1.3 }).data.slice();
    let within = 0, past = 0;
    for (let k = 0; k < n; k++) {
      const line = dist(k, c);
      if (line <= 1.25) { assert(Math.abs(reached[k] - line) <= 1e-9 * (1 + line), `within at ${k}`); ++within; }
      else if (line > 1.3) { assert(reached[k] === Infinity, `past at ${k}`); ++past; }
    }
    assert(within > 50 && past > 100, `within ${within}, past ${past}`);
    const nothing = tf.geodesicDistances(mesh, none).data.slice();
    assert(nothing.every((x) => x === Infinity), "no seed reaches nothing");
    both.delete(); centre.delete(); none.delete(); mesh.delete();
  });

  test("geodesicDistances: a piece no seed stands on is unreached", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4.0, 4.0, 4, 4, { dtype: "float64" });
    const f = plane.faces.data.slice(), p = plane.points.data.slice(), n = p.length / 3;
    const faces = new Int32Array(2 * f.length), points = new Float64Array(2 * p.length);
    faces.set(f); points.set(p);
    for (let k = 0; k < f.length; k++) faces[f.length + k] = f[k] + n;
    for (let k = 0; k < n; k++) {
      points[p.length + 3 * k] = p[3 * k] + 10;
      points[p.length + 3 * k + 1] = p[3 * k + 1];
      points[p.length + 3 * k + 2] = p[3 * k + 2];
    }
    plane.delete();
    const mesh = tf.mesh(faces, points);
    const seeds = ids(tf, [0]);
    const d = tf.geodesicDistances(mesh, seeds).data.slice();
    for (let k = 0; k < n; k++) assert(Number.isFinite(d[k]), `reached ${k}`);
    for (let k = n; k < 2 * n; k++) assert(d[k] === Infinity, `unreached ${k}`);
    seeds.delete(); mesh.delete();
  });

  test("geodesicDistances refuse at the boundary", () => {
    const tf = getTf();
    const mesh = tf.planeMesh(4.0, 4.0, 8, 8);
    const out = ids(tf, [mesh.numberOfPoints]), neg = ids(tf, [-1]), ok = ids(tf, [0]);
    const floats = values(tf, [0]);
    throws(() => tf.geodesicDistances(mesh, out), RangeError, "id past the end");
    throws(() => tf.geodesicDistances(mesh, neg), RangeError, "negative id");
    throws(() => tf.geodesicDistances(mesh, floats), TypeError, "float ids");
    throws(() => tf.geodesicDistances(mesh, ok, { radius: NaN }), RangeError, "NaN reach");
    throws(() => tf.geodesicDistances(mesh, ok, { radius: -1 }), RangeError, "negative reach");
    out.delete(); neg.delete(); ok.delete(); floats.delete(); mesh.delete();
  });

  for (const dtype of ["float32", "float64"]) {
    for (const order of ["harmonic", "biharmonic"]) {
      test(`dirichletField: the cotangent field on a plane is linear (${dtype}, ${order})`, () => {
        const tf = getTf();
        const mesh = tf.planeMesh(4.0, 4.0, 12, 12, { dtype });
        const linear = (x, y) => 2.0 * x - 3.0 * y + 0.5;
        const points = mesh.points.data.slice();
        const [held, vals] = rimHeld(points, linear, order === "biharmonic" ? 2 : 1, 12);
        assert(mesh.numberOfPoints - held.length > 50, "enough free points");
        const fi = ids(tf, held), fv = values(tf, vals);
        const result = tf.dirichletField(mesh, fi, fv, order);
        assert(result.status === tf.DirichletFieldStatus.solved, "solved");
        assert(result.field.dtype === dtype && result.field.length === mesh.numberOfPoints, "one value per point");
        const field = result.field.data.slice();
        // The field is stated in the mesh's dtype: a float32 one rounds to it.
        const tolerance = dtype === "float64" ? 1e-6 : 1e-5;
        for (let k = 0; k < mesh.numberOfPoints; k++)
          assert(Math.abs(field[k] - linear(points[3 * k], points[3 * k + 1])) < tolerance, `linear at ${k}`);
        result.field.delete(); fi.delete(); fv.delete(); mesh.delete();
      });
    }
  }

  for (const order of ["harmonic", "biharmonic"]) {
    test(`dirichletField: stated weights answer the dense solve (${order})`, () => {
      const tf = getTf();
      const cap = jitteredCap(tf);
      const mesh = tf.mesh(cap.faces, cap.points);
      const fi = ids(tf, cap.held), fv = values(tf, cap.vals);
      const w = tf.ndarray(cap.weights, [cap.weights.length]);
      const result = tf.dirichletField(mesh, fi, fv, order, { cornerWeights: w });
      assert(result.status === tf.DirichletFieldStatus.solved, "solved");
      const dense = denseField(cap, order === "biharmonic" ? 2 : 1);
      const field = result.field.data.slice();
      let worst = 0, spread = 0;
      for (let k = 0; k < dense.length; k++) {
        worst = Math.max(worst, Math.abs(field[k] - dense[k]));
        spread = Math.max(spread, Math.abs(dense[k] - 0.5));
      }
      assert(worst < 1e-10, `dense parity ${worst}`);
      assert(spread > 0.3, `spread ${spread}`);
      const cot = tf.dirichletField(mesh, fi, fv, order);
      let apart = 0;
      for (let k = 0; k < dense.length; k++) apart = Math.max(apart, Math.abs(cot.field.data.slice()[k] - field[k]));
      assert(apart > 0.05, `stated weights are read: ${apart}`);
      log(`  ${order}: dense parity ${worst.toExponential(2)}`, "line-pass");
      cot.field.delete(); result.field.delete(); w.delete(); fi.delete(); fv.delete(); mesh.delete();
    });
  }

  test("dirichletField: an unanchored run and indefinite weights refuse, holding the fixed rows", () => {
    const tf = getTf();
    const cap = jitteredCap(tf);
    const n = cap.points.length / 3;
    const faces = new Int32Array(cap.faces.length + 3);
    faces.set(cap.faces); faces.set([n, n + 1, n + 2], cap.faces.length);
    const points = new Float64Array(cap.points.length + 9);
    points.set(cap.points); points.set([4, 0, 0, 5, 0, 0, 4, 1, 0], cap.points.length);
    const weights = new Float64Array(cap.weights.length + 3);
    weights.set(cap.weights); weights.set([0.1, 0.3, 0.7], cap.weights.length);
    const island = tf.mesh(faces, points);
    const fi = ids(tf, cap.held), fv = values(tf, cap.vals);
    const w = tf.ndarray(weights, [weights.length]);
    for (const order of ["harmonic", "biharmonic"]) {
      const r = tf.dirichletField(island, fi, fv, order, { cornerWeights: w });
      assert(r.status === tf.DirichletFieldStatus.refusedUnanchored, `island refuses, ${order}`);
      assert(r.field.length === n + 3, "one value per point");
      cap.held.forEach((k, t) => assert(r.field.data.slice()[k] === cap.vals[t], "fixed row held"));
      r.field.delete();
    }
    const fiA = ids(tf, [...cap.held, n]), fvA = values(tf, [...cap.vals, 0.25]);
    const anchored = tf.dirichletField(island, fiA, fvA, "biharmonic", { cornerWeights: w });
    assert(anchored.status === tf.DirichletFieldStatus.solved, "anchoring the island solves");

    const mesh = tf.mesh(cap.faces, cap.points);
    const neg = tf.ndarray(cap.weights.map((x) => -x), [cap.weights.length]);
    const indefinite = tf.dirichletField(mesh, fi, fv, "harmonic", { cornerWeights: neg });
    assert(indefinite.status === tf.DirichletFieldStatus.refusedIndefinite, "indefinite weights refuse");
    cap.held.forEach((k, t) => assert(indefinite.field.data.slice()[k] === cap.vals[t], "fixed row held"));
    for (const x of [anchored.field, indefinite.field, w, neg, fi, fv, fiA, fvA, island, mesh]) x.delete();
  });

  test("dirichletField refuses at the boundary", () => {
    const tf = getTf();
    const mesh = tf.planeMesh(4.0, 4.0, 8, 8, { dtype: "float64" });
    const [held, vals] = rimHeld(mesh.points.data.slice(), (x) => x, 1, 8);
    const fi = ids(tf, held), fv = values(tf, vals);
    const twice = ids(tf, [held[0], held[0]]), pair = values(tf, [0, 1]);
    const short = values(tf, vals.slice(1)), inf = values(tf, vals.map(() => Infinity));
    const out = ids(tf, [mesh.numberOfPoints]), one = values(tf, [0]);
    const fewWeights = values(tf, [1, 1, 1]);
    const nanWeights = values(tf, new Array(3 * mesh.numberOfFaces).fill(NaN));
    throws(() => tf.dirichletField(mesh, twice, pair, "harmonic"), RangeError, "repeated id");
    throws(() => tf.dirichletField(mesh, fi, short, "harmonic"), RangeError, "one value per id");
    throws(() => tf.dirichletField(mesh, fi, inf, "harmonic"), RangeError, "finite values");
    throws(() => tf.dirichletField(mesh, fi, fv, "triharmonic"), RangeError, "order");
    throws(() => tf.dirichletField(mesh, out, one, "harmonic"), RangeError, "id past the end");
    throws(() => tf.dirichletField(mesh, fi, fv, "harmonic", { cornerWeights: fewWeights }), RangeError, "weight count");
    const unstated = tf.dirichletField(mesh, fi, fv, "harmonic", { cornerWeights: nanWeights });
    assert(unstated.status === tf.DirichletFieldStatus.refusedOperator, "weights that are not finite state no operator");
    const lonelyPoints = new Float64Array(mesh.points.data.slice().length + 3);
    lonelyPoints.set(mesh.points.data.slice()); lonelyPoints.set([9, 9, 9], mesh.points.data.slice().length);
    const lonely = tf.mesh(Int32Array.from(mesh.faces.data.slice()), lonelyPoints);
    const cornerless = tf.dirichletField(lonely, fi, fv, "harmonic");
    assert(cornerless.status === tf.DirichletFieldStatus.refusedOperator, "a point no face holds states no operator");
    for (const x of [unstated.field, cornerless.field, fi, fv, twice, pair, short, inf, out, one, fewWeights, nanWeights, lonely, mesh]) x.delete();
  });

  test("async: geodesicDistances and dirichletField match sync, and own their inputs", async () => {
    const tf = getTf();
    const cap = jitteredCap(tf);
    const mesh = tf.mesh(cap.faces, cap.points);
    const seeds = ids(tf, [0, 7]);
    const fi = ids(tf, cap.held), fv = values(tf, cap.vals);
    const w = tf.ndarray(cap.weights, [cap.weights.length]);
    const d = tf.geodesicDistances(mesh, seeds, { radius: 0.7 });
    const f1 = tf.dirichletField(mesh, fi, fv, "harmonic");
    const f2 = tf.dirichletField(mesh, fi, fv, "biharmonic", { cornerWeights: w });

    const pending = [
      tf.async.geodesicDistances(mesh, seeds, { radius: 0.7 }),
      tf.async.dirichletField(mesh, fi, fv, "harmonic"),
      tf.async.dirichletField(mesh, fi, fv, "biharmonic", { cornerWeights: w }),
    ];
    // The dispatched work captured every handle: releasing them now is safe.
    for (const x of [seeds, fi, fv, w, mesh]) x.delete();
    const [ad, af1, af2] = await Promise.all(pending);
    const same = (a, b) => a.length === b.length && a.every((x, k) => Object.is(x, b[k]));
    assert(same(ad.data.slice(), d.data.slice()), "async distances match sync");
    assert(af1.status === f1.status && same(af1.field.data.slice(), f1.field.data.slice()), "async cotangent field matches sync");
    assert(af2.status === f2.status && same(af2.field.data.slice(), f2.field.data.slice()), "async weighted field matches sync");
    for (const x of [d, f1.field, f2.field, ad, af1.field, af2.field]) x.delete();
  });

});

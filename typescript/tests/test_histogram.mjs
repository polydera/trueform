import { describe, test, log, assert, approx, getTf } from "./harness.mjs";

// ============================================================================
// bincount
// ============================================================================

describe("bincount", () => {

  test("basic", () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([1, 1, 2, 3, 3, 3]), [6]);
    const c = tf.bincount(x);
    assert(c.dtype === "int32", "bincount default dtype is int32");
    assert(c.shape[0] === 4, `shape [${c.shape}] expected [4]`);
    assert(c.data[0] === 0, "counts[0]");
    assert(c.data[1] === 2, "counts[1]");
    assert(c.data[2] === 1, "counts[2]");
    assert(c.data[3] === 3, "counts[3]");
    log("  [1,1,2,3,3,3] → [0,2,1,3]", "line-pass");
    c.delete(); x.delete();
  });

  test("minLength pads with zeros", () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([0, 1, 1]), [3]);
    const c = tf.bincount(x, { minLength: 6 });
    assert(c.shape[0] === 6, `shape [${c.shape}] expected [6]`);
    assert(c.data[0] === 1 && c.data[1] === 2
      && c.data[2] === 0 && c.data[3] === 0
      && c.data[4] === 0 && c.data[5] === 0, "padded");
    log("  minLength=6 pads zeros", "line-pass");
    c.delete(); x.delete();
  });

  test("output length is max(max(x)+1, minLength)", () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([5, 10]), [2]);
    const c = tf.bincount(x);
    assert(c.shape[0] === 11, `shape [${c.shape}] expected [11]`);
    assert(c.data[5] === 1 && c.data[10] === 1, "sparse");
    log("  [5,10] → length 11", "line-pass");
    c.delete(); x.delete();
  });

  test("weighted → float32", () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([0, 1, 1, 2]), [4]);
    const w = tf.ndarray(new Float32Array([0.5, 1.0, 2.0, 3.0]), [4]);
    const c = tf.bincount(x, { weights: w });
    assert(c.dtype === "float32", "weighted dtype is float32");
    assert(c.shape[0] === 3, `shape [${c.shape}]`);
    approx(c.data[0], 0.5, "bin 0");
    approx(c.data[1], 3.0, "bin 1");
    approx(c.data[2], 3.0, "bin 2");
    log("  weighted", "line-pass");
    c.delete(); w.delete(); x.delete();
  });

  test("single element", () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([5]), [1]);
    const c = tf.bincount(x);
    assert(c.shape[0] === 6, "length 6");
    for (let i = 0; i < 5; ++i) assert(c.data[i] === 0, `zero at ${i}`);
    assert(c.data[5] === 1, "one at 5");
    log("  [5] → [0,0,0,0,0,1]", "line-pass");
    c.delete(); x.delete();
  });

  test("empty input with minLength", () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([]), [0]);
    const c = tf.bincount(x, { minLength: 4 });
    assert(c.shape[0] === 4, `shape [${c.shape}] expected [4]`);
    for (let i = 0; i < 4; ++i) assert(c.data[i] === 0, `zero at ${i}`);
    log("  empty input respects minLength", "line-pass");
    c.delete(); x.delete();
  });

  test("negative value throws", () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([1, -1, 2]), [3]);
    let threw = false;
    try { const c = tf.bincount(x); c.delete(); }
    catch (e) { threw = true; }
    assert(threw, "should throw on negative input");
    log("  negative → throws", "line-pass");
    x.delete();
  });

  test("dtype coercion (float32 integers → int32)", () => {
    const tf = getTf();
    const x = tf.ndarray(new Float32Array([0, 1, 1, 2]), [4]);
    const c = tf.bincount(x);
    assert(c.shape[0] === 3, `shape [${c.shape}]`);
    assert(c.data[0] === 1 && c.data[1] === 2 && c.data[2] === 1, "counts");
    log("  float32 input coerced via .as(int32)", "line-pass");
    c.delete(); x.delete();
  });
});

// ============================================================================
// histogram
// ============================================================================

describe("histogram", () => {

  test("equal-width basic", () => {
    const tf = getTf();
    // 11 values evenly spaced in [0, 1]. 5 equal-width bins.
    // bins: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    // The value 1.0 lands in the last bin (inclusive upper edge).
    const x = tf.linspace(0, 1, 11);
    const { counts, edges } = tf.histogram(x, { bins: 5, range: [0, 1] });
    assert(counts.dtype === "int32", "counts default int32");
    assert(counts.shape[0] === 5, `counts shape [${counts.shape}]`);
    assert(edges.shape[0] === 6, `edges shape [${edges.shape}]`);
    let total = 0;
    for (let i = 0; i < 5; ++i) total += counts.data[i];
    assert(total === 11, `sum counts expected 11, got ${total}`);
    approx(edges.data[0], 0.0, "edge 0");
    approx(edges.data[5], 1.0, "edge last");
    log("  linspace(0,1,11) × 5 bins → sum 11", "line-pass");
    counts.delete(); edges.delete(); x.delete();
  });

  test("equal-width auto-range", () => {
    const tf = getTf();
    const x = tf.ndarray(new Float32Array([-1, 0, 1, 2, 3]), [5]);
    const { counts, edges } = tf.histogram(x, { bins: 4 });
    assert(edges.shape[0] === 5, "edges length 5");
    approx(edges.data[0], -1, "auto lo");
    approx(edges.data[4], 3, "auto hi");
    let total = 0;
    for (let i = 0; i < 4; ++i) total += counts.data[i];
    assert(total === 5, `all values counted, got ${total}`);
    log("  auto-range from data min/max", "line-pass");
    counts.delete(); edges.delete(); x.delete();
  });

  test("explicit edges", () => {
    const tf = getTf();
    // edges [0, 1, 3, 10] → 3 bins: [0,1), [1,3), [3,10]
    // x = [0.5, 2, 2.5, 9.9, 10]
    // bin 0 (x<1): 0.5 → 1
    // bin 1 (1<=x<3): 2, 2.5 → 2
    // bin 2 (3<=x<=10, inclusive last): 9.9, 10 → 2
    const x = tf.ndarray(new Float32Array([0.5, 2, 2.5, 9.9, 10]), [5]);
    const edges = tf.ndarray(new Float32Array([0, 1, 3, 10]), [4]);
    const r = tf.histogram(x, { bins: edges });
    assert(r.counts.shape[0] === 3, `counts shape [${r.counts.shape}]`);
    assert(r.counts.data[0] === 1, `bin 0 → ${r.counts.data[0]}`);
    assert(r.counts.data[1] === 2, `bin 1 → ${r.counts.data[1]}`);
    assert(r.counts.data[2] === 2, `bin 2 → ${r.counts.data[2]}`);
    log("  explicit edges with inclusive upper", "line-pass");
    r.counts.delete(); r.edges.delete(); edges.delete(); x.delete();
  });

  test("weighted → float32", () => {
    const tf = getTf();
    const x = tf.ndarray(new Float32Array([0.5, 1.5, 2.5]), [3]);
    const w = tf.ndarray(new Float32Array([10, 20, 30]), [3]);
    const { counts, edges } = tf.histogram(x,
      { bins: 3, range: [0, 3], weights: w });
    assert(counts.dtype === "float32", "counts dtype float32");
    approx(counts.data[0], 10, "bin 0");
    approx(counts.data[1], 20, "bin 1");
    approx(counts.data[2], 30, "bin 2");
    log("  weighted → float32 counts", "line-pass");
    counts.delete(); edges.delete(); w.delete(); x.delete();
  });

  test("density integrates to 1", () => {
    const tf = getTf();
    const x = tf.random("float32", [10000], -1, 1);
    const { counts, edges } = tf.histogram(x, {
      bins: 20, range: [-1, 1], density: true,
    });
    assert(counts.dtype === "float32", "density counts dtype");
    let integral = 0;
    for (let i = 0; i < 20; ++i)
      integral += counts.data[i] * (edges.data[i + 1] - edges.data[i]);
    approx(integral, 1.0, "density integral", 1e-3);
    log(`  density integral ≈ ${integral.toFixed(6)}`, "line-pass");
    counts.delete(); edges.delete(); x.delete();
  });

  test("NaN values dropped", () => {
    const tf = getTf();
    const clean = tf.ndarray(new Float32Array([0.1, 0.5, 0.9]), [3]);
    const withNan = tf.ndarray(
      new Float32Array([0.1, NaN, 0.5, NaN, 0.9]), [5]);
    const a = tf.histogram(clean, { bins: 4, range: [0, 1] });
    const b = tf.histogram(withNan, { bins: 4, range: [0, 1] });
    for (let i = 0; i < 4; ++i)
      assert(a.counts.data[i] === b.counts.data[i],
        `bin ${i}: ${a.counts.data[i]} vs ${b.counts.data[i]}`);
    log("  NaN silently dropped", "line-pass");
    a.counts.delete(); a.edges.delete();
    b.counts.delete(); b.edges.delete();
    clean.delete(); withNan.delete();
  });

  test("empty input returns zero counts", () => {
    const tf = getTf();
    const x = tf.ndarray(new Float32Array([]), [0]);
    const { counts, edges } = tf.histogram(x, { bins: 5, range: [0, 1] });
    assert(counts.shape[0] === 5, "counts shape");
    assert(edges.shape[0] === 6, "edges shape");
    for (let i = 0; i < 5; ++i) assert(counts.data[i] === 0, `zero ${i}`);
    log("  empty input → all zeros", "line-pass");
    counts.delete(); edges.delete(); x.delete();
  });

  test("tie on upper edge lands in last bin", () => {
    const tf = getTf();
    const x = tf.ndarray(new Float32Array([1.0]), [1]);
    const { counts, edges } = tf.histogram(x, { bins: 4, range: [0, 1] });
    assert(counts.data[3] === 1, "last bin gets tie");
    assert(counts.data[0] === 0 && counts.data[1] === 0
      && counts.data[2] === 0, "other bins empty");
    log("  x=1.0 with range [0,1] → last bin", "line-pass");
    counts.delete(); edges.delete(); x.delete();
  });

  test("balanced 0/1 array with 2 bins → 50/50", () => {
    const tf = getTf();
    // 500 zeros + 500 ones, interleaved. Edges for bins=2 over [0,1]
    // are [0.0, 0.5, 1.0]. Zeros land in bin 0; ones land in bin 1
    // (inclusive upper edge).
    const n = 1000;
    const data = new Float32Array(n);
    for (let i = 0; i < n; ++i) data[i] = i % 2;
    const x = tf.ndarray(data, [n]);
    const { counts, edges } = tf.histogram(x, { bins: 2, range: [0, 1] });
    assert(counts.shape[0] === 2, "2 bins");
    assert(counts.data[0] === 500,
      `bin 0: expected 500, got ${counts.data[0]}`);
    assert(counts.data[1] === 500,
      `bin 1: expected 500, got ${counts.data[1]}`);
    approx(edges.data[0], 0.0);
    approx(edges.data[1], 0.5);
    approx(edges.data[2], 1.0);
    log("  500×{0} + 500×{1} → [500, 500]", "line-pass");
    counts.delete(); edges.delete(); x.delete();
  });

  test("out-of-range values dropped", () => {
    const tf = getTf();
    const x = tf.ndarray(new Float32Array([-5, 0.5, 5]), [3]);
    const { counts, edges } = tf.histogram(x, { bins: 2, range: [0, 1] });
    let total = 0;
    for (let i = 0; i < 2; ++i) total += counts.data[i];
    assert(total === 1, `only 0.5 in range, got total ${total}`);
    log("  values outside [lo,hi] dropped", "line-pass");
    counts.delete(); edges.delete(); x.delete();
  });

  test("dtype matrix: default=int32, weights=float32, density=float32", () => {
    const tf = getTf();
    const x = tf.ndarray(new Float32Array([0.1, 0.5, 0.9]), [3]);
    const w = tf.ndarray(new Float32Array([1, 2, 3]), [3]);

    const a = tf.histogram(x, { bins: 3, range: [0, 1] });
    assert(a.counts.dtype === "int32", "default → int32");
    a.counts.delete(); a.edges.delete();

    const b = tf.histogram(x, { bins: 3, range: [0, 1], weights: w });
    assert(b.counts.dtype === "float32", "weights → float32");
    b.counts.delete(); b.edges.delete();

    const c = tf.histogram(x, { bins: 3, range: [0, 1], density: true });
    assert(c.counts.dtype === "float32", "density → float32");
    c.counts.delete(); c.edges.delete();

    log("  dtype promotion rules", "line-pass");
    w.delete(); x.delete();
  });

});

// ============================================================================
// async variants
// ============================================================================

describe("bincount (async)", () => {

  test("async basic matches sync", async () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([0, 1, 1, 2, 3, 3, 3]), [7]);
    const s = tf.bincount(x);
    const a = await tf.async.bincount(x);
    assert(a.shape[0] === s.shape[0], "shape match");
    for (let i = 0; i < s.shape[0]; ++i)
      assert(a.data[i] === s.data[i], `bin ${i}`);
    log("  async bincount matches sync", "line-pass");
    a.delete(); s.delete(); x.delete();
  });

  test("async weighted", async () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([0, 1, 1, 2]), [4]);
    const w = tf.ndarray(new Float32Array([0.5, 1, 2, 3]), [4]);
    const a = await tf.async.bincount(x, { weights: w });
    assert(a.dtype === "float32", "async weighted → float32");
    approx(a.data[0], 0.5); approx(a.data[1], 3.0); approx(a.data[2], 3.0);
    log("  async weighted", "line-pass");
    a.delete(); w.delete(); x.delete();
  });

  test("async negative rejected", async () => {
    const tf = getTf();
    const x = tf.ndarray(new Int32Array([1, -1, 2]), [3]);
    let threw = false;
    try {
      const r = await tf.async.bincount(x);
      r.delete();
    } catch (e) { threw = true; }
    assert(threw, "async throws on negative");
    log("  async negative → throws", "line-pass");
    x.delete();
  });
});

describe("histogram (async)", () => {

  test("async equal-width matches sync", async () => {
    const tf = getTf();
    const x = tf.linspace(0, 1, 11);
    const s = tf.histogram(x, { bins: 5, range: [0, 1] });
    const a = await tf.async.histogram(x, { bins: 5, range: [0, 1] });
    assert(a.counts.shape[0] === s.counts.shape[0], "counts shape");
    for (let i = 0; i < s.counts.shape[0]; ++i)
      assert(a.counts.data[i] === s.counts.data[i], `bin ${i}`);
    for (let i = 0; i < s.edges.shape[0]; ++i)
      approx(a.edges.data[i], s.edges.data[i], `edge ${i}`);
    log("  async equal-width matches sync", "line-pass");
    a.counts.delete(); a.edges.delete();
    s.counts.delete(); s.edges.delete();
    x.delete();
  });

  test("async explicit edges", async () => {
    const tf = getTf();
    const x = tf.ndarray(new Float32Array([0.5, 2, 2.5, 9.9, 10]), [5]);
    const edges = tf.ndarray(new Float32Array([0, 1, 3, 10]), [4]);
    const r = await tf.async.histogram(x, { bins: edges });
    assert(r.counts.data[0] === 1 && r.counts.data[1] === 2
      && r.counts.data[2] === 2, "bins match sync");
    log("  async explicit edges", "line-pass");
    r.counts.delete(); r.edges.delete(); edges.delete(); x.delete();
  });

  test("async weighted → float32", async () => {
    const tf = getTf();
    const x = tf.ndarray(new Float32Array([0.5, 1.5, 2.5]), [3]);
    const w = tf.ndarray(new Float32Array([10, 20, 30]), [3]);
    const r = await tf.async.histogram(x,
      { bins: 3, range: [0, 3], weights: w });
    assert(r.counts.dtype === "float32", "dtype");
    approx(r.counts.data[0], 10); approx(r.counts.data[1], 20);
    approx(r.counts.data[2], 30);
    log("  async weighted", "line-pass");
    r.counts.delete(); r.edges.delete(); w.delete(); x.delete();
  });

  test("async density integrates to 1", async () => {
    const tf = getTf();
    const x = tf.random("float32", [10000], -1, 1);
    const { counts, edges } = await tf.async.histogram(x, {
      bins: 20, range: [-1, 1], density: true,
    });
    let integral = 0;
    for (let i = 0; i < 20; ++i)
      integral += counts.data[i] * (edges.data[i + 1] - edges.data[i]);
    approx(integral, 1.0, "async density integral", 1e-3);
    log(`  async density integral ≈ ${integral.toFixed(6)}`, "line-pass");
    counts.delete(); edges.delete(); x.delete();
  });

  test("async large input sanity", async () => {
    const tf = getTf();
    const n = 1_000_000;
    const x = tf.random("float32", [n], -1, 1);
    const { counts, edges } = await tf.async.histogram(x, { bins: 100 });
    assert(counts.shape[0] === 100, "counts length");
    assert(edges.shape[0] === 101, "edges length");
    let total = 0;
    for (let i = 0; i < 100; ++i) total += counts.data[i];
    assert(total === n, `total expected ${n}, got ${total}`);
    log(`  async histogram of ${n} → sum ${total}`, "line-pass");
    counts.delete(); edges.delete(); x.delete();
  });
});

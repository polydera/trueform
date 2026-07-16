import { describe, test, log, assert, getTf } from "./harness.mjs";

const TOL = 1e-4;

function makeBoxes(dtype) {
  const tf = getTf();
  const opts = { dtype };
  const a = tf.boxMesh(1, 1, 1, undefined, undefined, undefined, opts);
  const b = tf.boxMesh(1, 1, 1, undefined, undefined, undefined, opts);
  b.points.add_(tf.ndarray(
    dtype === "float64" ? [0.4, 0.3, 0.2] : new Float32Array([0.4, 0.3, 0.2]),
    [3], opts,
  ));
  return [a, b];
}

function vol(mesh) {
  return Math.abs(getTf().signedVolume(mesh));
}

describe("csg: expressions", () => {
  test("builder chains produce postfix programs", () => {
    const tf = getTf();
    assert(JSON.stringify(tf.op(0).sub(tf.op(1)).program()) === "[0,1,-3]",
      "sub program");
    assert(JSON.stringify(tf.op(0).or(1).and(2).program()) === "[0,1,-1,2,-2]",
      "or/and with int promotion");
    assert(JSON.stringify(tf.op(0).not().program()) === "[0,-4]", "not");
  });
});

for (const dtype of ["float32", "float64"]) {
  describe(`csg: graph build + query (${dtype})`, () => {
    test("booleans match pairwise, one build many queries", () => {
      const tf = getTf();
      const [a, b] = makeBoxes(dtype);
      const graph = tf.csgGraph([a, b]);

      const diff = graph.mesh(tf.op(0).sub(tf.op(1)));
      const union = graph.mesh(tf.op(0).or(tf.op(1)));
      const inter = graph.mesh(tf.op(0).and(tf.op(1)));

      const ref = tf.booleanDifference(a, b);
      assert(Math.abs(vol(diff) - vol(ref.mesh)) < TOL, "difference volume");
      // inclusion-exclusion on the same build
      const bOnly = graph.mesh(tf.op(1).sub(tf.op(0)));
      assert(
        Math.abs(vol(union) - (vol(diff) + vol(bOnly) + vol(inter))) < TOL,
        "inclusion-exclusion");
      assert(Math.abs(vol(diff) + vol(inter) - 1.0) < TOL, "|A| = unit box");

      // remembered language-side state
      assert(graph.forms.length === 2 && graph.forms[0] === a, "forms");
      assert(graph.sheets.length === 0, "sheets");
      assert(graph.config.triangulation === "cdt", "config echo");

      const cp = graph.createdPoints;
      assert(cp.shape[1] === 3 && cp.shape[0] > 0, "createdPoints shape");
      assert(cp.dtype === dtype, "createdPoints dtype");

      for (const m of [diff, union, inter, bOnly, ref.mesh]) m.delete();
      ref.labels.delete();
      ref.faceLabels.delete();
      graph.delete();
      a.delete();
      b.delete();
    });

    test("labels and index maps resolve", () => {
      const tf = getTf();
      const [a, b] = makeBoxes(dtype);
      const graph = tf.csgGraph([a, b]);

      const lab = graph.mesh(tf.op(0).or(tf.op(1)), { returnSourceIds: true });
      const nFaces = lab.mesh.faces.shape[0];
      assert(lab.tagLabels.shape[0] === nFaces, "tagLabels parallel");
      assert(lab.faceLabels.shape[0] === nFaces, "faceLabels parallel");

      const im = graph.mesh(tf.op(0).or(tf.op(1)), { returnIndexMap: true });
      const nPts = im.mesh.points.shape[0];
      assert(im.pointTagLabels.shape[0] === nPts, "pointTagLabels parallel");
      assert(im.pointLabels.shape[0] === nPts, "pointLabels parallel");
      assert(im.nTags === 2, "nTags");
      assert(im.pointFOffsets.shape[0] === 3, "forward map offsets (nTags+1)");

      lab.mesh.delete(); lab.tagLabels.delete(); lab.faceLabels.delete();
      im.mesh.delete(); im.pointTagLabels.delete(); im.pointLabels.delete();
      im.faceTagLabels.delete(); im.faceLabels.delete();
      im.pointFOffsets.delete(); im.pointFData.delete();
      graph.delete(); a.delete(); b.delete();
    });

    test("domains partition the union", () => {
      const tf = getTf();
      const [a, b] = makeBoxes(dtype);
      const graph = tf.csgGraph([a, b]);

      const doms = graph.domains();
      assert(doms.meshes.length === 3, `3 domains, got ${doms.meshes.length}`);
      const union = graph.mesh(tf.op(0).or(tf.op(1)));
      const total = doms.meshes.reduce((s, m) => s + vol(m), 0);
      assert(Math.abs(total - vol(union)) < TOL, "domains sum to union");

      const inter = graph.domains(tf.op(0).and(tf.op(1)));
      assert(inter.meshes.length === 1, "expression-restricted domains");

      const labeled = graph.domains({ returnSourceIds: true });
      assert(labeled.tagOffsets.shape[0] === labeled.meshes.length + 1,
        "label blocks parallel to cells");

      for (const m of doms.meshes) m.delete();
      doms.ids.delete();
      for (const m of inter.meshes) m.delete();
      inter.ids.delete();
      for (const m of labeled.meshes) m.delete();
      labeled.ids.delete(); labeled.tagOffsets.delete();
      labeled.tagData.delete(); labeled.faceOffsets.delete();
      labeled.faceData.delete();
      union.delete();
      graph.delete(); a.delete(); b.delete();
    });

    test("full arrangement mesh (no expression)", () => {
      const tf = getTf();
      const [a, b] = makeBoxes(dtype);
      const graph = tf.csgGraph([a, b]);
      const full = graph.mesh();
      const union = graph.mesh(tf.op(0).or(tf.op(1)));
      assert(full.faces.shape[0] > union.faces.shape[0],
        "full arrangement has more faces than the union");
      const lab = graph.mesh({ returnSourceIds: true });
      assert(lab.tagLabels.shape[0] === lab.mesh.faces.shape[0],
        "no-expr labels parallel");
      let threw = false;
      try { graph.mesh({ returnIndexMap: true }); } catch { threw = true; }
      assert(threw, "index map without expression throws");
      full.delete(); union.delete();
      lab.mesh.delete(); lab.tagLabels.delete(); lab.faceLabels.delete();
      graph.delete(); a.delete(); b.delete();
    });

    test("every domains return path, with and without expression", () => {
      const tf = getTf();
      const [a, b] = makeBoxes(dtype);
      const graph = tf.csgGraph([a, b]);
      const e = tf.op(0).sub(tf.op(1));
      for (const expr of [undefined, e]) {
        const plain = expr ? graph.domains(expr) : graph.domains();
        const lab = expr
          ? graph.domains(expr, { returnSourceIds: true })
          : graph.domains({ returnSourceIds: true });
        const im = expr
          ? graph.domains(expr, { returnIndexMap: true })
          : graph.domains({ returnIndexMap: true });
        assert(plain.meshes.length === lab.meshes.length &&
               plain.meshes.length === im.meshes.length,
          "cell counts agree across paths");
        assert(lab.tagOffsets.shape[0] === plain.meshes.length + 1,
          "label offsets parallel");
        assert(im.pointTagOffsets.shape[0] === plain.meshes.length + 1,
          "index-map point offsets parallel");
        assert(im.nTags === 2, "index-map nTags");
        assert(im.inclusion.shape[0] === plain.meshes.length &&
               im.inclusion.shape[1] === 2, "inclusion [nCells, nTags]");
        // every kept cell lies inside some operand; entries are 0/1
        const inc = im.inclusion.data;
        for (let k = 0; k < im.inclusion.shape[0]; k++) {
          const inA = inc[2 * k], inB = inc[2 * k + 1];
          assert((inA === 0 || inA === 1) && (inB === 0 || inB === 1),
            "inclusion is 0/1");
          if (!expr) assert(inA || inB, "kept cells lie inside some operand");
        }
        for (const r of [plain, lab, im]) {
          for (const m of r.meshes) m.delete();
          r.ids.delete();
        }
        lab.tagOffsets.delete(); lab.tagData.delete();
        lab.faceOffsets.delete(); lab.faceData.delete();
        im.faceTagOffsets.delete(); im.faceTagData.delete();
        im.faceOffsets.delete(); im.faceData.delete();
        im.pointTagOffsets.delete(); im.pointTagData.delete();
        im.pointOffsets.delete(); im.pointData.delete();
        im.inclusion.delete();
      }
      graph.delete(); a.delete(); b.delete();
    });

    test("intersection curves", () => {
      const tf = getTf();
      const [a, b] = makeBoxes(dtype);
      const graph = tf.csgGraph([a, b]);
      const curves = graph.intersectionCurves();
      assert(curves.points.shape[0] > 0, "curve points exist");
      assert(curves.points.dtype === dtype, "curve dtype");
      curves.delete();
      graph.delete(); a.delete(); b.delete();
    });

    test("refined_cdt creates more points, stays closed", () => {
      const tf = getTf();
      const [a, b] = makeBoxes(dtype);
      const stock = tf.csgGraph([a, b]);
      const refined = tf.csgGraph([a, b], { triangulation: "refinedCdt" });
      assert(refined.config.triangulation === "refinedCdt", "config echo");
      assert(refined.createdPoints.shape[0] > stock.createdPoints.shape[0],
        "refined creates more points");
      const d0 = stock.mesh(tf.op(0).sub(tf.op(1)));
      const d1 = refined.mesh(tf.op(0).sub(tf.op(1)));
      assert(Math.abs(vol(d0) - vol(d1)) < TOL, "volumes agree across modes");
      d0.delete(); d1.delete();
      stock.delete(); refined.delete(); a.delete(); b.delete();
    });
  });
}

describe("csg: async", () => {
  test("async build + async queries agree with sync", async () => {
    const tf = getTf();
    const [a, b] = makeBoxes("float32");
    const graph = await tf.async.csgGraph([a, b]);
    const diff = await tf.async.csgMesh(graph, tf.op(0).sub(tf.op(1)));
    const doms = await tf.async.csgDomains(graph);
    const curves = await tf.async.csgIntersectionCurves(graph);
    assert(curves.points.shape[0] > 0, "async curves");
    curves.delete();
    const full = await tf.async.csgMesh(graph);
    assert(full.faces.shape[0] > 0, "async full arrangement");
    full.delete();
    const lab = await tf.async.csgMesh(graph, tf.op(0).or(1), { returnSourceIds: true });
    assert(lab.tagLabels.shape[0] === lab.mesh.faces.shape[0], "async labels");
    lab.mesh.delete(); lab.tagLabels.delete(); lab.faceLabels.delete();
    const im = await tf.async.csgMesh(graph, tf.op(0).or(1), { returnIndexMap: true });
    assert(im.nTags === 2, "async index map");
    im.mesh.delete(); im.pointTagLabels.delete(); im.pointLabels.delete();
    im.faceTagLabels.delete(); im.faceLabels.delete();
    im.pointFOffsets.delete(); im.pointFData.delete();
    const domsLab = await tf.async.csgDomains(graph, { returnSourceIds: true });
    assert(domsLab.tagOffsets.shape[0] === domsLab.meshes.length + 1, "async domain labels");
    for (const m of domsLab.meshes) m.delete();
    domsLab.ids.delete(); domsLab.tagOffsets.delete(); domsLab.tagData.delete();
    domsLab.faceOffsets.delete(); domsLab.faceData.delete();
    const syncGraph = tf.csgGraph([a, b]);
    const syncDiff = syncGraph.mesh(tf.op(0).sub(tf.op(1)));
    assert(Math.abs(vol(diff) - vol(syncDiff)) < TOL, "async == sync");
    assert(doms.meshes.length === 3, "async domains");
    diff.delete(); syncDiff.delete();
    for (const m of doms.meshes) m.delete();
    doms.ids.delete();
    graph.delete(); syncGraph.delete(); a.delete(); b.delete();
  });
});

// ============================================================================
// Outer shell
// ============================================================================

function twoOverlappingSpheres(dtype) {
  const tf = getTf();
  const opts = { dtype };
  const s0 = tf.sphereMesh(1, 16, 16, opts);
  const s1 = tf.sphereMesh(1, 16, 16, opts);
  // makeTranslation emits float32; the setter converts to the mesh dtype.
  s1.transformation = tf.makeTranslation(1, 0, 0);
  const merged = tf.concatenateMeshes([s0, s1]);
  return { tf, s0, s1, merged };
}

for (const dtype of ["float32", "float64"]) {
  describe(`csg: outerShell (${dtype})`, () => {
    test("repairs two overlapping spheres into a closed shell", () => {
      const { tf, s0, s1, merged } = twoOverlappingSpheres(dtype);

      const shell = tf.outerShell(merged);
      assert(shell.dtype === dtype, "dtype preserved");
      assert(shell.numberOfFaces > 0, "has faces");
      assert(tf.isClosed(shell), "shell is closed");
      const be = tf.boundaryEdges(shell);
      assert(be.shape[0] === 0, `0 boundary edges, got ${be.shape[0]}`);
      be.delete();
      const nm = tf.nonManifoldEdges(shell);
      assert(nm.shape[0] === 0, `0 non-manifold edges, got ${nm.shape[0]}`);
      nm.delete();

      const vShell = vol(shell);
      const vSphere = vol(s0);
      assert(vShell > vSphere + TOL,
        `shell volume ${vShell} exceeds single sphere ${vSphere}`);
      const ref = tf.booleanUnion(s0, s1);
      assert(Math.abs(vShell - vol(ref.mesh)) < 10 * TOL,
        `matches union volume: ${vShell} vs ${vol(ref.mesh)}`);
      log(`  shell: ${shell.numberOfFaces} faces, volume ${vShell.toFixed(4)}`,
        "line-pass");

      ref.mesh.delete(); ref.labels.delete(); ref.faceLabels.delete();
      shell.delete(); merged.delete(); s1.delete(); s0.delete();
    });

    test("ignoreOpenFragments accepted", () => {
      const { tf, s0, s1, merged } = twoOverlappingSpheres(dtype);
      const shell = tf.outerShell(merged, { ignoreOpenFragments: true });
      assert(tf.isClosed(shell), "shell is closed");
      shell.delete(); merged.delete(); s1.delete(); s0.delete();
    });
  });
}

describe("csg: outerShell (async)", () => {
  test("async matches sync", async () => {
    const { tf, s0, s1, merged } = twoOverlappingSpheres("float32");
    const shellSync = tf.outerShell(merged);
    const shellAsync = await tf.async.outerShell(merged);
    assert(tf.isClosed(shellAsync), "async shell is closed");
    assert(Math.abs(vol(shellAsync) - vol(shellSync)) < TOL,
      "async volume matches sync");
    shellAsync.delete(); shellSync.delete();
    merged.delete(); s1.delete(); s0.delete();
  });
});

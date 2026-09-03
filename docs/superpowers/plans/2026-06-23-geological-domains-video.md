# Geological Domains Pipeline Video — Implementation Plan

> **For agentic workers:** Execute task-by-task. Verification here is **visual** —
> render the artifact, Read the PNG, judge it — not unit tests. Slideshow-first:
> Tasks 1–5 produce static key-frames to lock the look; only Task 6+ adds motion.

**Goal:** A LinkedIn 4:5 portrait (1080×1350) video of the trueform pipeline running on
the 11-surface geological model — model assembling through load → arrange → curves →
(simplify) → domains see-through scan, with card-styled code + live stats beside it.

**Architecture:** VTK offscreen renders the evolving 3D model to PNG frames on the card
background; a card-style HTML+KaTeX overlay (title + code + stats) is screenshotted per
beat; PIL composites overlay over 3D per frame; imageio encodes mp4. A pipeline module
holds the trueform calls + stats so render and overlay share one source of truth.

**Tech Stack:** trueform (py venv `build-py3.11/.venv`), VTK 9.5, PIL, imageio[ffmpeg],
puppeteer (existing landing card flow), KaTeX.

**Spec:** `docs/superpowers/specs/2026-06-23-geological-domains-video-design.md`

---

## File structure

- `python/examples/vtk/geo_pipeline.py` — load the 11 OBJs, run the pipeline, return
  `arrangement`, `domain_labels`, `domains`, NM `paths`, and a `stats` dict (per-stage
  counts + ms). One source of truth for the numbers.
- `python/examples/vtk/geo_scene.py` — VTK scene: camera (z-up 3/4), arrangement actor
  with per-face RGBA ghost/highlight cell-data, NM-edge tube actor, domain palette;
  `render_state(...)` → RGB numpy frame for a given beat/highlight/camera.
- `scripts/geo-overlay.tmpl.html` (in landing `polydera-landing/scripts/`) + `build-geo-
  overlay.py` — card overlay (title + code lines + stat/time) → transparent PNG per beat.
- `python/examples/vtk/geo_compose.py` — composite (PIL) + slideshow/animation driver +
  imageio mp4 encode. Output to `article_drafts/`.

Constants shared (palette, card colors, canvas 1080×1350) live at the top of `geo_scene.py`.

---

### Task 1: Pipeline module + stats

**Files:** Create `python/examples/vtk/geo_pipeline.py`

- [ ] **Step 1: Implement `run_pipeline(input_dir)`**

```python
import os, glob, time
import numpy as np
import trueform as tf

DT, IDT = np.float64, np.int32

def _t(fn):
    t0 = time.perf_counter(); out = fn(); return out, (time.perf_counter() - t0) * 1000

def run_pipeline(input_dir, do_simplify=True):
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(input_dir), "*.obj")))
    meshes, in_tris = [], 0
    t0 = time.perf_counter()
    for p in paths:
        f, pt = tf.read_obj(p, ngon=3, dtype=DT, index_dtype=IDT)
        meshes.append(tf.Mesh(f, pt)); in_tris += len(f)
    load_ms = (time.perf_counter() - t0) * 1000

    ((af, ap), _tags, _f), arr_ms = _t(lambda: tf.mesh_arrangements(meshes))
    (afc_apc), clean_ms = _t(lambda: tf.cleaned((af, ap), tolerance=1e-6))
    af, ap = afc_apc
    arrangement = tf.Mesh(af, ap)

    nm, nm_ms = _t(lambda: tf.non_manifold_edges(arrangement))
    nm_paths = tf.connect_edges_to_paths(nm) if len(nm) > 0 else []

    simp = None; simp_ms = 0.0
    if do_simplify:
        (sf_sp), simp_ms = _t(lambda: tf.simplified(
            (af, ap), error_rel=0.002, min_quality=0.3, feature_angle=60.0,
            iterations=3, optimize_iterations=3))
        simp = sf_sp

    dl, dl_ms = _t(lambda: tf.domain_labels(arrangement, ignore_open_fragments=True))
    (comps_ids), split_ms = _t(lambda: tf.split_into_domains(arrangement, dl))
    comps, ids = comps_ids

    stats = {
        "n_files": len(paths), "in_tris": in_tris, "load_ms": load_ms,
        "arr_tris": len(af), "arr_pts": len(ap), "arr_ms": arr_ms, "clean_ms": clean_ms,
        "nm_edges": len(nm), "nm_polylines": len(nm_paths), "nm_ms": nm_ms,
        "simp_tris": (len(simp[0]) if simp else 0), "simp_ms": simp_ms,
        "n_domains": int(dl[1]), "dl_ms": dl_ms, "split_ms": split_ms,
    }
    return {"meshes": meshes, "arrangement": arrangement, "dl": dl,
            "domains": comps, "domain_labels": ids, "nm_paths": nm_paths,
            "nm_points": arrangement.points, "stats": stats}
```

- [ ] **Step 2: Verify the numbers**

Run: `cd python/examples/vtk && /Users/ziga/trueform/build-py3.11/.venv/bin/python -c "import geo_pipeline,json; print(json.dumps(geo_pipeline.run_pipeline('~/Downloads/inputs')['stats'], indent=1))"`
Expected: `in_tris=489242`, `arr_tris=587982`, `nm_edges=24979`, `nm_polylines=283`,
`n_domains=90`. If any differ, the pipeline call is wrong — fix before proceeding.

---

### Task 2: VTK scene + first still (beat: arrange)

**Files:** Create `python/examples/vtk/geo_scene.py`

- [ ] **Step 1: Scene constants + helpers**

Card palette + canvas; reuse `numpy_to_polydata` from `util`. Domain `PALETTE` and the
`apply_highlight` / `make_tubes_actor` mechanics copied from `domains_from_files.py`
(per-face RGBA cell-data; tubes via `tf.make_tube_mesh`, radius `0.001 * aabb_diagonal`).
Card bg gradient `SetBackground(0.031,0.055,0.047)` / `SetBackground2(0.055,0.090,0.082)`.

- [ ] **Step 2: `render_state(pipe, beat, highlight=None, size=(1080,1080))` → RGB ndarray**

Builds the renderer for the given beat (arrangement actor; tubes for `curves`/`domains`;
ghost+highlight cell-data when `highlight` is a domain label), fixed z-up 3/4 camera
framed on the arrangement bounds, offscreen, returns the RGB frame (via
`vtkWindowToImageFilter`, flipped). Camera: `SetViewUp(0,0,1)`, position from the AABB
(`pos = center + R*(0.6,-1.5,0.55)` scaled to diagonal), `ResetCamera` then `Zoom(~0.9)`.

- [ ] **Step 3: Render the arrange still**

Run a one-off: `render_state(pipe, "arrange")` → `imageio.imwrite("/tmp/geo_arrange.png", frame)`.

- [ ] **Step 4: Verify framing**

Read `/tmp/geo_arrange.png`. The geological block fills the frame, z-up, readable 3/4
angle, card bg. Adjust camera azimuth/elevation/zoom until it reads well. (This is the
camera-lock checkpoint.)

---

### Task 3: Card overlay (title + code + stats)

**Files:** Create `polydera-landing/scripts/geo-overlay.tmpl.html` + `build-geo-overlay.py`

- [ ] **Step 1: HTML/CSS overlay template (portrait 1080×1350, transparent)**

Card vars (`--bg` transparent here; title band + bottom code/stats card). Title:
`Volumetric Domains` (teal/rose) + `geological model` mono subtitle. Bottom card: 1–2
mono code lines with teal `tf.` calls, and a stat/time line (KaTeX numbers). A
`__TITLE__ / __CODE__ / __STAT__` placeholder set, filled per beat.

- [ ] **Step 2: `build-geo-overlay.py` — emit one overlay PNG per beat**

Reads a per-beat dict (title, code lines, stat string), injects into the template,
screenshots the `.overlay` element transparent via the existing `render-card.mjs`
puppeteer flow (deviceScaleFactor 1, exact 1080×1350), writes `overlay_<beat>.png`.

- [ ] **Step 3: Verify**

Build the `arrange` overlay; Read `overlay_arrange.png`. Title + code + stat read in card
style, transparent bg, correct portrait size.

---

### Task 4: Composite — first full key-frame (DESIGN LOCK)

**Files:** Create `python/examples/vtk/geo_compose.py`

- [ ] **Step 1: `compose(frame_3d, overlay_png, layout)` (PIL)**

Portrait layout: card bg canvas 1080×1350; paste the 3D frame into the hero rect
(title band top ~135px, hero ~840px, code strip bottom ~375px); paste the transparent
overlay full-canvas on top. Return the composited RGB image.

- [ ] **Step 2: Build the `arrange` key-frame**

3D `render_state(pipe,"arrange")` + `overlay_arrange.png` → `compose` →
`/tmp/geo_keyframe_arrange.png`.

- [ ] **Step 3: Verify — judge the whole look**

Read `/tmp/geo_keyframe_arrange.png`. This is the **design-lock checkpoint**: proportions
(title/hero/code split), card colors on dark bg, code formatting, camera. Iterate here
until it's right — everything else inherits this.

---

### Task 5: All 5 beat key-frames → the slideshow

**Files:** Modify `geo_compose.py` (driver)

- [ ] **Step 1: Per-beat content table**

Beats `load / arrange / curves / simplify / domains` with their code lines + stat strings
pulled from `pipe["stats"]` (e.g. `arrange`: `arr = tf.mesh_arrangements(meshes)` /
`587,982 △ · 50 ms`). `domains` beat uses `render_state(..., highlight=<a named domain>)`
with the see-through arrangement.

- [ ] **Step 2: Render all 5 key-frames**

Loop beats → `render_state` + overlay + `compose` → `/tmp/geo_kf_<beat>.png`.

- [ ] **Step 3: Verify — review the slideshow**

Read all 5. Confirm the narrative reads as stills. **Decisions taken here:** simplify
beat in/out; final camera; domain palette/ghost alpha on dark bg; any copy tweaks.
STOP and get user sign-off on the stills before motion.

---

### Task 6: Motion + encode (after slideshow sign-off)

**Files:** Modify `geo_compose.py`

- [ ] **Step 1: Per-beat animation**

- load: surfaces fade in (alpha ramp). arrange: hold. curves: tubes fade/grow in.
  simplify: crossfade arr→simplified. domains: arrangement → alpha 25%, then **scan** —
  sort domains by centroid z ascending, step `highlight` through them (dwell on named,
  fast-sweep rest). Camera gentle drift beats 1–3, static during the scan.
- Each frame: `render_state` (cached per unchanged state) + the beat overlay (with a
  sub-step for when the stat appears) → `compose`.

- [ ] **Step 2: Encode**

`imageio.get_writer("article_drafts/geo_domains_linkedin.mp4", fps=30, codec="libx264",
quality=8, macro_block_size=8)`; append composited frames; target < 40 s.

- [ ] **Step 3: Verify**

Read sampled frames (early/mid/scan/end) + confirm 1080×1350, h264, duration. Send the
mp4 for review.

---

## Self-review

- **Spec coverage:** load/arrange/curves/simplify/domains beats → Tasks 1,5,6; portrait
  composition → Task 4; see-through scan → Task 6; card overlay → Task 3; slideshow-first
  → Tasks 1–5 gate before Task 6. Covered.
- **Consistency:** `run_pipeline` keys (`stats`, `domains`, `nm_paths`) are the same names
  used in Tasks 2/5/6. `render_state(pipe, beat, highlight, size)` signature stable.
- **Deferred (intentional):** simplify in/out and exact camera/pacing resolve on the
  Task 4–5 stills, per spec.

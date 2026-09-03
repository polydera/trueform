# Geological Domains Pipeline Video — Design

**Goal:** A short, card-styled "watch trueform build it" video on the real 11-surface
geological model: the model assembles through the actual pipeline (load → arrange →
curves → simplify → domains), with the real code calls + live stats/times shown
beside it, ending in a see-through scan through the 90 volumetric domains.

**Primary deliverable:** the **LinkedIn 4:5 portrait (1080×1350)** cut. YouTube 16:9
(1920×1080) is a follow-on with the same content, re-laid-out (not a crop).

**Aesthetic:** the card system — dark teal bg `#0c1513`, teal `#00d5be` / rose
`#c44569` accents, Inter + mono, KaTeX for the numbers. Cell/domain palette as in
`domains_from_files.py`.

---

## Data (real, already validated)

11 files in `~/Downloads/inputs` (D1–D4 deposits, Dyke 1/2 veins, GnD/QzP intrusions,
OVB erosion, No Material, bounding box). UTM coords, z-up, extent 1395×1190×823 m,
AABB diagonal ~2010. Pipeline numbers (matching the paper's geological figure):

| stage | call | result | time |
|---|---|---|---|
| load | `read_obj × 11` | 489,242 △ | 38 ms |
| arrange | `mesh_arrangements(meshes)` | 587,982 △, 271k pts | 50 ms |
| clean | `cleaned(.., 1e-6)` | 587,820 △ | 9 ms |
| curves | `non_manifold_edges → connect_edges_to_paths` | 24,979 edges → 283 polylines | 6 ms |
| simplify *(optional beat)* | `simplified(.., error_rel=0.002)` | 151,500 △ (25.8%) | 573 ms |
| domains | `domain_labels → split_into_domains` | 90 domains | 9 ms + 13 ms |

## Composition (portrait 4:5)

Vertical stack, full card background:
- **Title band** (top, ~10%): `Volumetric Domains` (teal/rose) · `geological model` mono subtitle.
- **3D hero** (~62%): the evolving model, card dark bg, z-up 3/4 camera.
- **Code + stats strip** (bottom, ~28%): the current pipeline call(s) in card mono
  with light syntax color (teal for `tf.` calls), and the stat/time appearing under
  it as the call "runs" (KaTeX numbers). One or two lines visible at a time, advancing
  per beat — a mini live-notebook.

(YouTube 16:9 later: same parts, side-by-side — code/stats panel left ~40%, 3D right.)

## Narrative — 5 beats

1. **Load.** 11 surfaces fade in, each colored by geological layer. Code: `read_obj × 11`
   → *489,242 △ · 38 ms*. Gentle camera drift.
2. **Arrange.** Surfaces co-refine into one mesh (color settles toward the arrangement).
   Code: `mesh_arrangements` → *587,982 △ · 50 ms*.
3. **Curves.** The intersection network lights up as NM-edge tubes (radius 0.1% of
   diagonal). Code: `non_manifold_edges → paths` → *24,979 edges → 283 polylines · 6 ms*.
4. **Simplify** *(optional, decide at slideshow review).* Arrangement coarsens, topology
   intact. Code: `simplified(0.002)` → *151,500 △ (25.8%) · 573 ms*.
5. **Domains — see-through scan.** Arrangement goes translucent (alpha ~25%), NM tubes
   stay crisp as the skeleton. Sort the 90 domains **bottom→top by centroid z** and
   **scroll** through them: each domain lights up opaque in its color while the rest stay
   ghosted (the `apply_highlight` mechanic from `domains_from_files.py`, automated).
   Camera holds still during the scan so the eye tracks the lit domain. Dwell briefly on
   the named geological domains, fast-sweep the rest. Code: `domain_labels →
   split_into_domains` → *90 domains*.

Single see-through view for portrait (no separate extracted-domain panel — that's a
YouTube-landscape addition).

## Technical approach

- **3D**: VTK offscreen, render each frame to PNG (RGB) on the card bg. Reuse the
  mechanics from `domains_from_files.py` / `domains.py`: arrangement actor with per-face
  RGBA cell-data for ghost/highlight, NM-edge tubes via `make_tube_mesh`, domain palette.
- **Overlay** (title + code + stats): rendered as **HTML + KaTeX** in the card style
  (same flow as the cards: a template → puppeteer screenshot, transparent PNG), one
  overlay state per beat (and per sub-step where stats appear).
- **Composite**: PIL — overlay PNG over the 3D frame per frame.
- **Encode**: `imageio` (bundled ffmpeg), H.264 mp4, 30 fps, 1080×1350.
- **Self-contained Python** for the render/composite/encode (HTML overlay generated via
  the existing puppeteer card flow).

## Slideshow-first

Before any motion, render the **5 beat key-frames** as static composites (3D still +
final overlay state for that beat) at 1080×1350. These lock: layout proportions, card
colors, code formatting/syntax, camera framing, domain palette on the dark bg. Review
and adjust the *look* on the stills; only then add motion (fades, the curve light-up,
the domain scan) and transitions.

## Camera

z-up, 3/4 view of the block (front-ish from −y, elevated). Gentle drift in beats 1–3,
static during the domain scan (beat 5). Tune exact angle on the first slideshow still.

## Open / deferred

- **Simplify beat** in or out — decide on the slideshow review (it's a clean "shrinks
  but stays valid" beat but adds time; optional).
- **Length / pacing** — target < 40 s for LinkedIn; finalize after the key-frames.
- **YouTube 16:9 cut** — after the LinkedIn version ships.

## Files

- New: `python/examples/vtk/geo_video.py` (or a small package) — pipeline + VTK frame
  render + scan.
- New: overlay HTML template + a build script under the landing `scripts/` card flow.
- New: a compositor/encoder script.
- Output: `article_drafts/geo_domains_linkedin.mp4` (+ key-frame stills).

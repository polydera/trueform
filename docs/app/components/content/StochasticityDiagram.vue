<script setup lang="ts">
import katex from "katex";
import "katex/dist/katex.min.css";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

const renderMath = (tex: string) =>
  katex.renderToString(tex, { throwOnError: false, displayMode: false });

// Math expressions
const meshPrime = renderMath(
  "\\mathcal{M}':\\left\\{\\;\\begin{smallmatrix} T' \\\\ G' \\end{smallmatrix}\\;\\right\\}"
);
const mesh = renderMath(
  "\\mathcal{M}:\\left\\{\\;\\begin{smallmatrix} T' \\cup T_\\Delta \\\\ G' + G_\\Delta \\end{smallmatrix}\\;\\right\\}"
);
const topArrow = renderMath("\\cup\\, \\textcolor{#14b8a6}{T_\\Delta} \\sim \\mathcal{D}_T");
const bottomArrow = renderMath("+\\, \\textcolor{#14b8a6}{G_\\Delta} \\sim \\mathcal{D}_G");
const fArrow = renderMath("\\mathcal{F}");

// Caption math
const capMPrime = renderMath("\\mathcal{M}'");
const capM = renderMath("\\mathcal{M}");
const capTDelta = renderMath("\\textcolor{#14b8a6}{T_\\Delta}");
const capGDelta = renderMath("\\textcolor{#14b8a6}{G_\\Delta}");
const capF = renderMath("\\mathcal{F}");

// SVG viewBox
const VB = { x: 0, y: 60, w: 420, h: 175 };

// --- Mesh geometry (unchanged) ---
const gridSize = 4;
const cellSize = 16;
const isoX = (x: number, y: number) => (x - y) * 0.866 * cellSize;
const isoY = (x: number, y: number) => (x + y) * 0.5 * cellSize;

const cleanGridPoints: { x: number; y: number }[][] = [];
for (let i = 0; i <= gridSize; i++) {
  cleanGridPoints[i] = [];
  for (let j = 0; j <= gridSize; j++) cleanGridPoints[i][j] = { x: isoX(i, j), y: isoY(i, j) };
}

const corruptedGridPoints: { x: number; y: number }[][] = [];
for (let i = 0; i <= gridSize; i++) {
  corruptedGridPoints[i] = [];
  for (let j = 0; j <= gridSize; j++) {
    if (i === 1 && j === 1) {
      corruptedGridPoints[i][j] = { x: isoX(i - 0.4, j - 0.4), y: isoY(i - 0.4, j - 0.4) };
    } else {
      corruptedGridPoints[i][j] = { x: isoX(i, j), y: isoY(i, j) };
    }
  }
}

const generateGridPath = (points: { x: number; y: number }[][]) => {
  let path = "";
  for (let i = 0; i <= gridSize; i++) {
    path += `M ${points[i][0].x} ${points[i][0].y} `;
    for (let j = 1; j <= gridSize; j++) path += `L ${points[i][j].x} ${points[i][j].y} `;
  }
  for (let j = 0; j <= gridSize; j++) {
    path += `M ${points[0][j].x} ${points[0][j].y} `;
    for (let i = 1; i <= gridSize; i++) path += `L ${points[i][j].x} ${points[i][j].y} `;
  }
  return path;
};

const cleanPath = generateGridPath(cleanGridPoints);
const corruptedPath = generateGridPath(corruptedGridPoints);

const flapHeight = 20;
const flapTilt = -10;
const v1 = corruptedGridPoints[2][3];
const v2 = corruptedGridPoints[4][3];
const v3 = { x: v2.x + flapTilt, y: v2.y - flapHeight + flapTilt * 0.5 };
const v4 = { x: v1.x + flapTilt, y: v1.y - flapHeight * 0.85 + flapTilt * 0.5 };
const flapPath = `M ${v1.x} ${v1.y} L ${v2.x} ${v2.y} L ${v3.x} ${v3.y} L ${v4.x} ${v4.y} Z`;

const vMid = corruptedGridPoints[3][3];
const vMidTop = { x: (v3.x + v4.x) / 2, y: (v3.y + v4.y) / 2 };
const flapLines = `M ${vMid.x} ${vMid.y} L ${vMidTop.x} ${vMidTop.y}`;

const isVertexBehindFlap = (i: number, j: number) => (i === 1 && j === 2) || (i === 2 && j === 2);

const displacedVertex = corruptedGridPoints[1][1];
const displacedEdges = `
  M ${corruptedGridPoints[0][1].x} ${corruptedGridPoints[0][1].y} L ${displacedVertex.x} ${displacedVertex.y}
  M ${corruptedGridPoints[2][1].x} ${corruptedGridPoints[2][1].y} L ${displacedVertex.x} ${displacedVertex.y}
  M ${corruptedGridPoints[1][0].x} ${corruptedGridPoints[1][0].y} L ${displacedVertex.x} ${displacedVertex.y}
  M ${corruptedGridPoints[1][2].x} ${corruptedGridPoints[1][2].y} L ${displacedVertex.x} ${displacedVertex.y}
`;
const isDisplacedVertex = (i: number, j: number) => i === 1 && j === 1;

// --- HTML overlay positioning ---
const svgEl = ref<SVGSVGElement | null>(null);

type PxMap = { left: number; top: number; sx: number; sy: number };
const pxMap = ref<PxMap>({ left: 0, top: 0, sx: 1, sy: 1 });

const updateMap = () => {
  const svg = svgEl.value;
  if (!svg) return;
  const r = svg.getBoundingClientRect();
  pxMap.value = {
    left: r.left,
    top: r.top,
    sx: r.width / VB.w,
    sy: r.height / VB.h,
  };
};

let ro: ResizeObserver | null = null;
const onWin = () => updateMap();

onMounted(() => {
  updateMap();
  window.addEventListener("resize", onWin, { passive: true });
  if (svgEl.value && "ResizeObserver" in window) {
    ro = new ResizeObserver(() => updateMap());
    ro.observe(svgEl.value);
  }
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", onWin);
  if (ro && svgEl.value) ro.unobserve(svgEl.value);
  ro = null;
});

// Convert viewBox coordinates to overlay pixel styles.
// `x,y,w,h` are in SVG viewBox coordinates.
const boxStyle = (x: number, y: number, w: number, h: number) => {
  const m = pxMap.value;
  // y in SVG is relative to viewBox y-origin (60). Our SVG uses viewBox "0 60 420 175".
  const pxLeft = (x - VB.x) * m.sx;
  const pxTop = (y - VB.y) * m.sy;
  return {
    left: `${pxLeft}px`,
    top: `${pxTop}px`,
    width: `${w * m.sx}px`,
    height: `${h * m.sy}px`,
  };
};

// These correspond to your original foreignObject placements.
const overlay = computed(() => {
  return {
    // Titles (these were inside g translate + negative x/y; global ends up at x=5,y=70 and x=285,y=70)
    meshPrime: boxStyle(5, 70, 130, 55),
    mesh: boxStyle(285, 70, 140, 55),

    // Arrow labels (already in global SVG coords)
    topArrow: boxStyle(170, 165, 110, 25),
    bottomArrow: boxStyle(170, 198, 110, 25),

    // F label
    fArrow: boxStyle(200, 85, 50, 30),
  };
});
</script>

<template>
  <div class="stochasticity-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <div class="svg-wrap">
        <svg ref="svgEl" viewBox="0 60 420 175" class="w-full">
          <!-- Clean mesh (left) -->
          <g transform="translate(70, 160)">
            <path :d="cleanPath" fill="none" stroke="currentColor" stroke-width="1.2" />
            <template v-for="i in gridSize + 1" :key="'row-' + i">
              <template v-for="j in gridSize + 1" :key="'col-' + j">
                <circle
                  :cx="cleanGridPoints[i - 1][j - 1].x"
                  :cy="cleanGridPoints[i - 1][j - 1].y"
                  r="2.5"
                  fill="currentColor"
                />
              </template>
            </template>
          </g>

          <!-- Corrupted mesh (right) -->
          <g transform="translate(350, 160)">
            <path :d="corruptedPath" fill="none" stroke="currentColor" stroke-width="1.2" />
            <path :d="displacedEdges" fill="none" class="displaced-stroke" />
            <path :d="flapPath" class="flap-fill" />
            <path :d="flapLines" class="flap-fill" fill="none" />
            <template v-for="i in gridSize + 1" :key="'row-c-' + i">
              <template v-for="j in gridSize + 1" :key="'col-c-' + j">
                <circle
                  :cx="corruptedGridPoints[i - 1][j - 1].x"
                  :cy="corruptedGridPoints[i - 1][j - 1].y"
                  r="2.5"
                  :fill="isDisplacedVertex(i - 1, j - 1) ? 'rgb(20, 184, 166)' : 'currentColor'"
                  :opacity="isVertexBehindFlap(i - 1, j - 1) ? 0.15 : 1"
                />
              </template>
            </template>
          </g>

          <!-- Bottom arrow -->
          <path
            d="M 145 168 C 185 205, 255 205, 295 168"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            marker-end="url(#arrowhead)"
          />

          <!-- Top arrow -->
          <path
            d="M 295 140 C 255 105, 185 105, 145 140"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            marker-end="url(#arrowhead)"
          />

          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
            </marker>
          </defs>
        </svg>

        <!-- HTML overlay (KaTeX) -->
        <div class="overlay" aria-hidden="true">
          <div class="kbox kbox-lg" :style="overlay.meshPrime" v-html="meshPrime"></div>
          <div class="kbox kbox-lg" :style="overlay.mesh" v-html="mesh"></div>

          <div class="kbox kbox-sm" :style="overlay.topArrow" v-html="topArrow"></div>
          <div class="kbox kbox-sm" :style="overlay.bottomArrow" v-html="bottomArrow"></div>

          <div class="kbox kbox-lg" :style="overlay.fArrow" v-html="fArrow"></div>
        </div>
      </div>
    </div>

    <p class="diagram-caption">
      Processing adds artifacts to the underlying mesh <span v-html="capMPrime" />: topological noise
      <span v-html="capTDelta" /> (flaps, bad winding) and geometric noise <span v-html="capGDelta" />
      (displaced vertices). Idealization <span v-html="capF" /> recovers <span v-html="capMPrime" />
      from <span v-html="capM" />.
    </p>
  </div>
</template>

<style scoped>
.stochasticity-diagram {
  color: var(--color-neutral-900);
  flex-direction: column;
  align-items: center;
}
.dark .stochasticity-diagram {
  color: var(--color-neutral-100);
}

.diagram-box {
  background: rgba(0, 0, 0, 0.03);
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  padding: 1.25rem 2rem;
  max-width: 38rem;
  width: 100%;
}
.dark .diagram-box {
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(255, 255, 255, 0.1);
}

.svg-wrap {
  position: relative;
  width: 100%;
}

/* Overlay sits on top of the SVG */
.overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

/* KaTeX boxes, centered like your original flex containers */
.kbox {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  transform: translateZ(0); /* helps Safari compositing */
}
.kbox-lg {
  font-size: clamp(0.8rem, 2.5vw, 1.2rem);
}
.kbox-sm {
  font-size: clamp(0.55rem, 1.8vw, 0.75rem);
}

/* KaTeX color inherits from component */
.kbox :deep(.katex) {
  color: inherit;
}

.flap-fill {
  fill: rgba(20, 184, 166, 0.85);
  stroke: rgb(15, 118, 110);
  stroke-width: 2.5;
}
.displaced-stroke {
  stroke: rgb(20, 184, 166);
  stroke-width: 2;
}

.diagram-caption {
  text-align: center;
  font-size: 0.85rem;
  color: var(--color-neutral-500);
  font-style: italic;
  margin-top: 0.75rem;
  max-width: 38rem;
  line-height: 1.5;
}
.dark .diagram-caption {
  color: var(--color-neutral-400);
}
.diagram-caption :deep(.katex) {
  font-style: normal;
  color: inherit;
}
</style>


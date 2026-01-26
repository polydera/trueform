<script setup lang="ts">
import katex from "katex";
import "katex/dist/katex.min.css";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

const renderMath = (tex: string) =>
  katex.renderToString(tex, {
    throwOnError: false,
    displayMode: false,
  });

// Math labels for diagram
const mathB = renderMath("\\mathcal{B}");
const mathF = renderMath("\\mathcal{F}");
const mathG = renderMath("\\mathcal{G}_\\mathcal{I}");

// Math labels for caption
const capB = renderMath("\\mathcal{B}");
const capF = renderMath("\\mathcal{F}");
const capG = renderMath("\\mathcal{G}_\\mathcal{I}");

// SVG viewBox
const VB = { x: 0, y: 0, w: 520, h: 340 };

// --- HTML overlay positioning (Safari-safe; no foreignObject) ---
const svgEl = ref<SVGSVGElement | null>(null);

type PxMap = { sx: number; sy: number };
const pxMap = ref<PxMap>({ sx: 1, sy: 1 });

const updateMap = () => {
  const svg = svgEl.value;
  if (!svg) return;
  const r = svg.getBoundingClientRect();
  pxMap.value = {
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

// Convert viewBox coordinates to overlay pixel box.
// `x,y,w,h` are in SVG viewBox units.
const boxStyle = (x: number, y: number, w: number, h: number) => {
  const m = pxMap.value;
  return {
    left: `${x * m.sx}px`,
    top: `${y * m.sy}px`,
    width: `${w * m.sx}px`,
    height: `${h * m.sy}px`,
  };
};

/**
 * Original label placements (global SVG coords):
 *
 * Top row group: translate(0,20)
 *  - Arrow B group translate(175,60) => origin (175,80). FO: x=5, y=-30, w=30, h=25 => (180,50)
 *  - Dashed->G group translate(345,60) => origin (345,80). FO: x=-2, y=-25, w=40, h=25 => (343,55)
 *
 * Vertical F arrows:
 *  - left: g translate(90,150). FO x=8,y=8,w=30,h=30 => (98,158)
 *  - right: g translate(280,150). FO x=8,y=8,w=30,h=30 => (288,158)
 *
 * Bottom row group: translate(0,200)
 *  - Arrow B group translate(175,60) => origin (175,260). FO x=5,y=-30 => (180,230)
 *  - Dashed->G group translate(345,60) => origin (345,260). FO x=-2,y=-25 => (343,235)
 */
const overlay = computed(() => {
  return {
    // Top row labels
    topB: boxStyle(180, 50, 30, 25),
    topG: boxStyle(343, 51, 40, 25),

    // Middle vertical F labels
    leftF: boxStyle(98, 158, 30, 30),
    rightF: boxStyle(288, 158, 30, 30),

    // Bottom row labels
    bottomB: boxStyle(180, 230, 30, 25),
    bottomG: boxStyle(343, 231, 40, 25),
  };
});
</script>

<template>
  <div class="commutation-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <div class="svg-wrap">
        <svg ref="svgEl" viewBox="0 0 520 340" class="w-full">
          <!-- Top row: corrupted inputs -->
          <g transform="translate(0, 20)">
            <!-- Sphere with artifact + Cylinder box -->
            <g transform="translate(20, 0)">
              <rect x="0" y="0" width="140" height="120" rx="8" class="inner-box" />
              <!-- Sphere with flap -->
              <g transform="translate(35, 60)">
                <circle cx="0" cy="0" r="28" class="shape-fill" />
                <circle cx="0" cy="0" r="28" class="shape-stroke" fill="none" />
                <ellipse cx="0" cy="0" rx="28" ry="10" class="shape-stroke" fill="none" />
                <path d="M 22 0 L 38 -4 L 38 20 L 22 16 Z" class="flap-fill" />
              </g>
              <!-- Cylinder -->
              <g transform="translate(105, 60)">
                <ellipse cx="0" cy="-20" rx="28" ry="10" class="shape-fill" />
                <rect x="-28" y="-20" width="56" height="40" class="shape-fill" />
                <ellipse cx="0" cy="20" rx="28" ry="10" class="shape-fill" />
                <line x1="-28" y1="-20" x2="-28" y2="20" class="shape-stroke" />
                <line x1="28" y1="-20" x2="28" y2="20" class="shape-stroke" />
                <ellipse cx="0" cy="-20" rx="28" ry="10" class="shape-stroke" fill="none" />
                <ellipse cx="0" cy="20" rx="28" ry="10" class="shape-stroke" fill="none" />
              </g>
            </g>

            <!-- Arrow B (no foreignObject) -->
            <g transform="translate(175, 60)">
              <line x1="0" y1="0" x2="40" y2="0" class="arrow-line" marker-end="url(#arrowhead)" />
            </g>

            <!-- Union result with artifact -->
            <g transform="translate(230, 0)">
              <rect x="0" y="0" width="100" height="120" rx="8" class="inner-box" />
              <g transform="translate(50, 60)">
                <path d="M -28 0 A 28 28 0 0 0 28 0" class="shape-fill" />
                <ellipse cx="0" cy="-30" rx="28" ry="10" class="shape-fill" />
                <rect x="-28" y="-30" width="56" height="30" class="shape-fill" />
                <line x1="-28" y1="-30" x2="-28" y2="0" class="shape-stroke" />
                <line x1="28" y1="-30" x2="28" y2="0" class="shape-stroke" />
                <ellipse cx="0" cy="-30" rx="28" ry="10" class="shape-stroke" fill="none" />
                <path d="M -28 0 A 28 28 0 0 0 28 0" class="shape-stroke" fill="none" />
                <path d="M 22 0 L 38 -4 L 38 20 L 22 16 Z" class="flap-fill" />
              </g>
            </g>

            <!-- Dashed line to graph (no foreignObject) -->
            <g transform="translate(345, 60)">
              <line x1="0" y1="0" x2="35" y2="0" class="dashed-line" />
            </g>

            <!-- Intersection graph with branching -->
            <g transform="translate(395, 0)">
              <rect x="0" y="0" width="105" height="120" rx="8" class="inner-box" />
              <g transform="translate(52, 60)">
                <ellipse cx="0" cy="0" rx="40" ry="22" class="graph-stroke" fill="none" />
                <circle cx="-40" cy="0" r="3" class="graph-node" />
                <circle cx="40" cy="0" r="3" class="graph-node" />
                <circle cx="0" cy="-22" r="3" class="graph-node" />
                <circle cx="0" cy="22" r="3" class="graph-node" />
                <circle cx="-28" cy="-16" r="3" class="graph-node" />
                <circle cx="28" cy="-16" r="3" class="graph-node" />
                <circle cx="-28" cy="16" r="3" class="graph-node" />
                <line x1="28" y1="16" x2="28" y2="-6" class="flap-stroke" />
                <circle cx="28" cy="-6" r="3" class="flap-node" />
                <circle cx="28" cy="16" r="3" class="flap-node" />
              </g>
            </g>
          </g>

          <!-- Vertical F arrow (left) (no foreignObject) -->
          <g transform="translate(90, 150)">
            <line x1="0" y1="0" x2="0" y2="40" class="arrow-line" marker-end="url(#arrowhead)" />
          </g>

          <!-- Vertical F arrow (right) (no foreignObject) -->
          <g transform="translate(280, 150)">
            <line x1="0" y1="0" x2="0" y2="40" class="arrow-line" marker-end="url(#arrowhead)" />
          </g>

          <!-- Bottom row: idealized inputs -->
          <g transform="translate(0, 200)">
            <!-- Clean sphere + Cylinder box -->
            <g transform="translate(20, 0)">
              <rect x="0" y="0" width="140" height="120" rx="8" class="inner-box" />
              <g transform="translate(35, 60)">
                <circle cx="0" cy="0" r="28" class="shape-fill" />
                <circle cx="0" cy="0" r="28" class="shape-stroke" fill="none" />
                <ellipse cx="0" cy="0" rx="28" ry="10" class="shape-stroke" fill="none" />
              </g>
              <g transform="translate(105, 60)">
                <ellipse cx="0" cy="-20" rx="28" ry="10" class="shape-fill" />
                <rect x="-28" y="-20" width="56" height="40" class="shape-fill" />
                <ellipse cx="0" cy="20" rx="28" ry="10" class="shape-fill" />
                <line x1="-28" y1="-20" x2="-28" y2="20" class="shape-stroke" />
                <line x1="28" y1="-20" x2="28" y2="20" class="shape-stroke" />
                <ellipse cx="0" cy="-20" rx="28" ry="10" class="shape-stroke" fill="none" />
                <ellipse cx="0" cy="20" rx="28" ry="10" class="shape-stroke" fill="none" />
              </g>
            </g>

            <!-- Arrow B (no foreignObject) -->
            <g transform="translate(175, 60)">
              <line x1="0" y1="0" x2="40" y2="0" class="arrow-line" marker-end="url(#arrowhead)" />
            </g>

            <!-- Clean union result -->
            <g transform="translate(230, 0)">
              <rect x="0" y="0" width="100" height="120" rx="8" class="inner-box" />
              <g transform="translate(50, 60)">
                <path d="M -28 0 A 28 28 0 0 0 28 0" class="shape-fill" />
                <ellipse cx="0" cy="-30" rx="28" ry="10" class="shape-fill" />
                <rect x="-28" y="-30" width="56" height="30" class="shape-fill" />
                <line x1="-28" y1="-30" x2="-28" y2="0" class="shape-stroke" />
                <line x1="28" y1="-30" x2="28" y2="0" class="shape-stroke" />
                <ellipse cx="0" cy="-30" rx="28" ry="10" class="shape-stroke" fill="none" />
                <path d="M -28 0 A 28 28 0 0 0 28 0" class="shape-stroke" fill="none" />
              </g>
            </g>

            <!-- Dashed line to graph (no foreignObject) -->
            <g transform="translate(345, 60)">
              <line x1="0" y1="0" x2="35" y2="0" class="dashed-line" />
            </g>

            <!-- Clean intersection graph -->
            <g transform="translate(395, 0)">
              <rect x="0" y="0" width="105" height="120" rx="8" class="inner-box" />
              <g transform="translate(52, 60)">
                <ellipse cx="0" cy="0" rx="40" ry="22" class="graph-stroke" fill="none" />
                <circle cx="-40" cy="0" r="3" class="graph-node" />
                <circle cx="40" cy="0" r="3" class="graph-node" />
                <circle cx="0" cy="-22" r="3" class="graph-node" />
                <circle cx="0" cy="22" r="3" class="graph-node" />
                <circle cx="-28" cy="-16" r="3" class="graph-node" />
                <circle cx="28" cy="-16" r="3" class="graph-node" />
                <circle cx="-28" cy="16" r="3" class="graph-node" />
                <circle cx="28" cy="16" r="3" class="graph-node" />
              </g>
            </g>
          </g>

          <!-- Arrowhead marker -->
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
            </marker>
          </defs>
        </svg>

        <!-- HTML overlay (KaTeX) -->
        <div class="overlay" aria-hidden="true">
          <!-- Top row labels -->
          <div class="kbox kbox-lg" :style="overlay.topB" v-html="mathB"></div>
          <div class="kbox kbox-sm" :style="overlay.topG" v-html="mathG"></div>

          <!-- Vertical F labels -->
          <div class="kbox kbox-lg" :style="overlay.leftF" v-html="mathF"></div>
          <div class="kbox kbox-lg" :style="overlay.rightF" v-html="mathF"></div>

          <!-- Bottom row labels -->
          <div class="kbox kbox-lg" :style="overlay.bottomB" v-html="mathB"></div>
          <div class="kbox kbox-sm" :style="overlay.bottomG" v-html="mathG"></div>
        </div>
      </div>
    </div>

    <p class="diagram-caption">
      Boolean operation <span v-html="capB" /> commutes with mesh idealization <span v-html="capF" />.
      <strong>Top:</strong> <span v-html="capB" /> on a mesh with artifacts produces a result with artifacts;
      the intersection graph <span v-html="capG" /> contains a branching edge.
      <strong>Bottom:</strong> <span v-html="capB" /> on idealized meshes produces a clean result;
      <span v-html="capG" /> is a simple closed curve. Apply <span v-html="capF" /> at any point—both paths
      yield equivalent geometry.
    </p>
  </div>
</template>

<style scoped>
.commutation-diagram {
  color: var(--color-neutral-900);
}

.dark .commutation-diagram {
  color: var(--color-neutral-100);
}

.diagram-box {
  background: rgba(0, 0, 0, 0.03);
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  max-width: 42rem;
  width: 100%;
}

.dark .diagram-box {
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(255, 255, 255, 0.1);
}

/* Wrap SVG + overlay */
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

/* KaTeX boxes */
.kbox {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  transform: translateZ(0); /* helps Safari compositing */
}
.kbox-lg {
  font-size: clamp(0.95rem, 2.5vw, 1.1rem);
}
.kbox-sm {
  font-size: clamp(0.75rem, 2vw, 0.9rem);
}
.kbox :deep(.katex) {
  color: inherit;
}

.inner-box {
  fill: rgba(0, 0, 0, 0.04);
  stroke: rgba(0, 0, 0, 0.1);
  stroke-width: 1;
}

.dark .inner-box {
  fill: rgba(255, 255, 255, 0.06);
  stroke: rgba(255, 255, 255, 0.1);
}

.shape-fill {
  fill: rgba(0, 0, 0, 0.06);
  stroke: none;
}

.dark .shape-fill {
  fill: rgba(255, 255, 255, 0.1);
}

.shape-stroke {
  stroke: currentColor;
  stroke-width: 1;
}

.flap-fill {
  fill: rgba(20, 184, 166, 0.85);
  stroke: rgb(15, 118, 110);
  stroke-width: 1.5;
}

.flap-node {
  fill: rgb(20, 184, 166);
  stroke: rgb(15, 118, 110);
  stroke-width: 1;
}

.flap-stroke {
  stroke: rgb(20, 184, 166);
  stroke-width: 1.5;
}

.graph-stroke {
  stroke: currentColor;
  stroke-width: 1.2;
}

.graph-node {
  fill: currentColor;
}

.arrow-line {
  stroke: currentColor;
  stroke-width: 1.5;
}

.dashed-line {
  stroke: currentColor;
  stroke-width: 1.2;
  stroke-dasharray: 4 3;
}

.diagram-caption {
  text-align: center;
  font-size: 0.85rem;
  color: var(--color-neutral-500);
  font-style: italic;
  margin-top: 0.75rem;
  max-width: 42rem;
  line-height: 1.5;
}

.dark .diagram-caption {
  color: var(--color-neutral-400);
}
</style>


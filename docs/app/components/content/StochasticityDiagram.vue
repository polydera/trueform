<script setup lang="ts">
import katex from "katex";
import "katex/dist/katex.min.css";

const renderMath = (tex: string) => {
  return katex.renderToString(tex, {
    throwOnError: false,
    displayMode: false
  });
};

// Math expressions - M: with stacked T,G inside braces, more spacing
const meshPrime = renderMath("\\mathcal{M}':\\left\\{\\;\\begin{smallmatrix} T' \\\\ G' \\end{smallmatrix}\\;\\right\\}");
const mesh = renderMath("\\mathcal{M}:\\left\\{\\;\\begin{smallmatrix} T' \\cup T_\\Delta \\\\ G' + G_\\Delta \\end{smallmatrix}\\;\\right\\}");
const topArrow = renderMath("\\cup\\, \\textcolor{#14b8a6}{T_\\Delta} \\sim \\mathcal{D}_T");
const bottomArrow = renderMath("+\\, \\textcolor{#14b8a6}{G_\\Delta} \\sim \\mathcal{D}_G");
const fArrow = renderMath("\\mathcal{F}");

// Caption math
const capMPrime = renderMath("\\mathcal{M}'");
const capM = renderMath("\\mathcal{M}");
const capTDelta = renderMath("\\textcolor{#14b8a6}{T_\\Delta}");
const capGDelta = renderMath("\\textcolor{#14b8a6}{G_\\Delta}");
const capF = renderMath("\\mathcal{F}");

// Generate isometric grid points for a 5x5 mesh
// Isometric projection: x' = (x - y) * cos(30°), y' = (x + y) * sin(30°)
const gridSize = 4;
const cellSize = 16;
const isoX = (x: number, y: number) => (x - y) * 0.866 * cellSize;
const isoY = (x: number, y: number) => (x + y) * 0.5 * cellSize;

// Clean grid points
const cleanGridPoints: { x: number; y: number }[][] = [];
for (let i = 0; i <= gridSize; i++) {
  cleanGridPoints[i] = [];
  for (let j = 0; j <= gridSize; j++) {
    cleanGridPoints[i][j] = { x: isoX(i, j), y: isoY(i, j) };
  }
}

// Corrupted grid points (one vertex displaced)
const corruptedGridPoints: { x: number; y: number }[][] = [];
for (let i = 0; i <= gridSize; i++) {
  corruptedGridPoints[i] = [];
  for (let j = 0; j <= gridSize; j++) {
    if (i === 1 && j === 1) {
      // Displaced vertex
      corruptedGridPoints[i][j] = { x: isoX(i - 0.4, j - 0.4), y: isoY(i - 0.4, j - 0.4) };
    } else {
      corruptedGridPoints[i][j] = { x: isoX(i, j), y: isoY(i, j) };
    }
  }
}

// Generate grid lines as path
const generateGridPath = (points: { x: number; y: number }[][]) => {
  let path = "";
  // Horizontal lines
  for (let i = 0; i <= gridSize; i++) {
    path += `M ${points[i][0].x} ${points[i][0].y} `;
    for (let j = 1; j <= gridSize; j++) {
      path += `L ${points[i][j].x} ${points[i][j].y} `;
    }
  }
  // Vertical lines
  for (let j = 0; j <= gridSize; j++) {
    path += `M ${points[0][j].x} ${points[0][j].y} `;
    for (let i = 1; i <= gridSize; i++) {
      path += `L ${points[i][j].x} ${points[i][j].y} `;
    }
  }
  return path;
};

const cleanPath = generateGridPath(cleanGridPoints);
const corruptedPath = generateGridPath(corruptedGridPoints);

// Flap artifact on corrupted mesh - attached to grid vertices
// Flap goes from vertices (2,3) to (4,3) - two cells wide, tilted forward
const flapHeight = 20;
const flapTilt = -10; // backward tilt offset
const v1 = corruptedGridPoints[2][3]; // vertex at grid position (2,3)
const v2 = corruptedGridPoints[4][3]; // vertex at grid position (4,3)
const v3 = { x: v2.x + flapTilt, y: v2.y - flapHeight + flapTilt * 0.5 }; // top right, tilted forward
const v4 = { x: v1.x + flapTilt, y: v1.y - flapHeight * 0.85 + flapTilt * 0.5 }; // top left, tilted forward

const flapPath = `M ${v1.x} ${v1.y} L ${v2.x} ${v2.y} L ${v3.x} ${v3.y} L ${v4.x} ${v4.y} Z`;
// Middle line on flap for grid effect (at the middle vertex position)
const vMid = corruptedGridPoints[3][3];
const vMidTop = { x: (v3.x + v4.x) / 2, y: (v3.y + v4.y) / 2 };
const flapLines = `M ${vMid.x} ${vMid.y} L ${vMidTop.x} ${vMidTop.y}`;

// Vertices that are visually behind the flap (should be dimmed)
const isVertexBehindFlap = (i: number, j: number) => {
  return (i === 1 && j === 2) || (i === 2 && j === 2);
};

// Displaced vertex and its 4 connected edges (for highlighting geometric noise)
const displacedVertex = corruptedGridPoints[1][1];
const displacedEdges = `
  M ${corruptedGridPoints[0][1].x} ${corruptedGridPoints[0][1].y} L ${displacedVertex.x} ${displacedVertex.y}
  M ${corruptedGridPoints[2][1].x} ${corruptedGridPoints[2][1].y} L ${displacedVertex.x} ${displacedVertex.y}
  M ${corruptedGridPoints[1][0].x} ${corruptedGridPoints[1][0].y} L ${displacedVertex.x} ${displacedVertex.y}
  M ${corruptedGridPoints[1][2].x} ${corruptedGridPoints[1][2].y} L ${displacedVertex.x} ${displacedVertex.y}
`;

// Check if vertex is the displaced one
const isDisplacedVertex = (i: number, j: number) => i === 1 && j === 1;
</script>

<template>
  <div class="stochasticity-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <svg viewBox="0 60 420 175" class="w-full">
      <!-- Left side: M': {...} -->
      <g transform="translate(70, 95)">
        <foreignObject x="-65" y="-25" width="130" height="55">
          <div class="math-container" v-html="meshPrime" />
        </foreignObject>
      </g>

      <!-- Right side: M: {...} -->
      <g transform="translate(350, 95)">
        <foreignObject x="-65" y="-25" width="140" height="55">
          <div class="math-container" v-html="mesh" />
        </foreignObject>
      </g>

      <!-- Clean mesh (left) -->
      <g transform="translate(70, 160)">
        <path :d="cleanPath" fill="none" stroke="currentColor" stroke-width="1.2" />
        <!-- Vertex dots -->
        <template v-for="i in gridSize + 1" :key="'row-' + i">
          <template v-for="j in gridSize + 1" :key="'col-' + j">
            <circle
              :cx="cleanGridPoints[i-1][j-1].x"
              :cy="cleanGridPoints[i-1][j-1].y"
              r="2.5"
              fill="currentColor"
            />
          </template>
        </template>
      </g>

      <!-- Corrupted mesh (right) -->
      <g transform="translate(350, 160)">
        <path :d="corruptedPath" fill="none" stroke="currentColor" stroke-width="1.2" />
        <!-- Highlighted displaced edges (geometric noise) -->
        <path :d="displacedEdges" fill="none" class="displaced-stroke" />
        <!-- Flap artifact -->
        <path :d="flapPath" class="flap-fill" />
        <path :d="flapLines" class="flap-fill" fill="none" />
        <!-- Vertex dots -->
        <template v-for="i in gridSize + 1" :key="'row-c-' + i">
          <template v-for="j in gridSize + 1" :key="'col-c-' + j">
            <circle
              :cx="corruptedGridPoints[i-1][j-1].x"
              :cy="corruptedGridPoints[i-1][j-1].y"
              r="2.5"
              :fill="isDisplacedVertex(i-1, j-1) ? 'rgb(20, 184, 166)' : 'currentColor'"
              :opacity="isVertexBehindFlap(i-1, j-1) ? 0.15 : 1"
            />
          </template>
        </template>
      </g>

      <!-- Arrow from left mesh to right mesh (bottom path) -->
      <path d="M 145 168 C 185 205, 255 205, 295 168" fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#arrowhead)" />

      <!-- Labels on bottom arrow - separate positions -->
      <foreignObject x="170" y="172" width="110" height="25">
        <div class="math-container-small" v-html="topArrow" />
      </foreignObject>
      <foreignObject x="170" y="198" width="110" height="25">
        <div class="math-container-small" v-html="bottomArrow" />
      </foreignObject>

      <!-- Arrow from right back to left (top path with F) -->
      <path d="M 295 140 C 255 105, 185 105, 145 140" fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#arrowhead)" />

      <!-- F label on top arrow -->
      <foreignObject x="200" y="85" width="50" height="30">
        <div class="math-container" v-html="fArrow" />
      </foreignObject>

      <!-- Arrowhead marker -->
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
        </marker>
      </defs>
      </svg>
    </div>
    <p class="diagram-caption">
      Processing adds artifacts to the underlying mesh <span v-html="capMPrime" />: topological noise <span v-html="capTDelta" /> (flaps, bad winding) and geometric noise <span v-html="capGDelta" /> (displaced vertices). Idealization <span v-html="capF" /> recovers <span v-html="capMPrime" /> from <span v-html="capM" />.
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

.math-container {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.05rem;
}

.math-container-small {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
}

.math-container :deep(.katex),
.math-container-small :deep(.katex) {
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

.flap-text {
  color: rgb(13, 148, 136);
  font-weight: 500;
}
</style>

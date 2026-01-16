<script setup lang="ts">
import katex from "katex";
import "katex/dist/katex.min.css";

const renderMath = (tex: string) => {
  return katex.renderToString(tex, {
    throwOnError: false,
    displayMode: false
  });
};

// Math labels for diagram
const mathB = renderMath("\\mathcal{B}");
const mathF = renderMath("\\mathcal{F}");
const mathG = renderMath("\\mathcal{G}_\\mathcal{I}");

// Math labels for caption
const capB = renderMath("\\mathcal{B}");
const capF = renderMath("\\mathcal{F}");
const capG = renderMath("\\mathcal{G}_\\mathcal{I}");
</script>

<template>
  <div class="commutation-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <svg viewBox="0 0 520 340" class="w-full">
        <!-- Top row: corrupted inputs -->
        <g transform="translate(0, 20)">
          <!-- Sphere with artifact + Cylinder box -->
          <g transform="translate(20, 0)">
            <rect x="0" y="0" width="140" height="120" rx="8" class="inner-box" />
            <!-- Sphere with flap -->
            <g transform="translate(35, 60)">
              <circle cx="0" cy="0" r="28" class="shape-fill" />
              <circle cx="0" cy="0" r="28" class="shape-stroke" fill="none" />
              <!-- Equator line for 3D hint -->
              <ellipse cx="0" cy="0" rx="28" ry="10" class="shape-stroke" fill="none" />
              <!-- Flap artifact (top at equator) -->
              <path d="M 22 0 L 38 -4 L 38 20 L 22 16 Z" class="flap-fill" />
            </g>
            <!-- Cylinder (same radius 28 as sphere) -->
            <g transform="translate(105, 60)">
              <ellipse cx="0" cy="-20" rx="28" ry="10" class="shape-fill" />
              <rect x="-28" y="-20" width="56" height="40" class="shape-fill" />
              <ellipse cx="0" cy="20" rx="28" ry="10" class="shape-fill" />
              <!-- Outline -->
              <line x1="-28" y1="-20" x2="-28" y2="20" class="shape-stroke" />
              <line x1="28" y1="-20" x2="28" y2="20" class="shape-stroke" />
              <ellipse cx="0" cy="-20" rx="28" ry="10" class="shape-stroke" fill="none" />
              <ellipse cx="0" cy="20" rx="28" ry="10" class="shape-stroke" fill="none" />
            </g>
          </g>

          <!-- Arrow B -->
          <g transform="translate(175, 60)">
            <line x1="0" y1="0" x2="40" y2="0" class="arrow-line" marker-end="url(#arrowhead)" />
            <foreignObject x="5" y="-30" width="30" height="25">
              <div class="math-label" v-html="mathB" />
            </foreignObject>
          </g>

          <!-- Union result with artifact -->
          <g transform="translate(230, 0)">
            <rect x="0" y="0" width="100" height="120" rx="8" class="inner-box" />
            <g transform="translate(50, 60)">
              <!-- Hemisphere bottom (sphere radius 28) -->
              <path d="M -28 0 A 28 28 0 0 0 28 0" class="shape-fill" />
              <!-- Cylinder on top (same radius 28) -->
              <ellipse cx="0" cy="-30" rx="28" ry="10" class="shape-fill" />
              <rect x="-28" y="-30" width="56" height="30" class="shape-fill" />
              <!-- Outline -->
              <line x1="-28" y1="-30" x2="-28" y2="0" class="shape-stroke" />
              <line x1="28" y1="-30" x2="28" y2="0" class="shape-stroke" />
              <ellipse cx="0" cy="-30" rx="28" ry="10" class="shape-stroke" fill="none" />
              <path d="M -28 0 A 28 28 0 0 0 28 0" class="shape-stroke" fill="none" />
              <!-- Flap on result (top at junction) -->
              <path d="M 22 0 L 38 -4 L 38 20 L 22 16 Z" class="flap-fill" />
            </g>
          </g>

          <!-- Dashed line to graph -->
          <g transform="translate(345, 60)">
            <line x1="0" y1="0" x2="35" y2="0" class="dashed-line" />
            <foreignObject x="-2" y="-25" width="40" height="25">
              <div class="math-label-small" v-html="mathG" />
            </foreignObject>
          </g>

          <!-- Intersection graph with branching -->
          <g transform="translate(395, 0)">
            <rect x="0" y="0" width="105" height="120" rx="8" class="inner-box" />
            <g transform="translate(52, 60)">
              <ellipse cx="0" cy="0" rx="40" ry="22" class="graph-stroke" fill="none" />
              <!-- Nodes on ellipse -->
              <circle cx="-40" cy="0" r="3" class="graph-node" />
              <circle cx="40" cy="0" r="3" class="graph-node" />
              <circle cx="0" cy="-22" r="3" class="graph-node" />
              <circle cx="0" cy="22" r="3" class="graph-node" />
              <circle cx="-28" cy="-16" r="3" class="graph-node" />
              <circle cx="28" cy="-16" r="3" class="graph-node" />
              <circle cx="-28" cy="16" r="3" class="graph-node" />
              <!-- Bottom right vertex is artifact with edge going up -->
              <line x1="28" y1="16" x2="28" y2="-6" class="flap-stroke" />
              <circle cx="28" cy="-6" r="3" class="flap-node" />
              <circle cx="28" cy="16" r="3" class="flap-node" />
            </g>
          </g>
        </g>

        <!-- Vertical F arrow (left) -->
        <g transform="translate(90, 150)">
          <line x1="0" y1="0" x2="0" y2="40" class="arrow-line" marker-end="url(#arrowhead)" />
          <foreignObject x="8" y="8" width="30" height="30">
            <div class="math-label" v-html="mathF" />
          </foreignObject>
        </g>

        <!-- Vertical F arrow (right) -->
        <g transform="translate(280, 150)">
          <line x1="0" y1="0" x2="0" y2="40" class="arrow-line" marker-end="url(#arrowhead)" />
          <foreignObject x="8" y="8" width="30" height="30">
            <div class="math-label" v-html="mathF" />
          </foreignObject>
        </g>

        <!-- Bottom row: idealized inputs -->
        <g transform="translate(0, 200)">
          <!-- Clean sphere + Cylinder box -->
          <g transform="translate(20, 0)">
            <rect x="0" y="0" width="140" height="120" rx="8" class="inner-box" />
            <!-- Clean sphere -->
            <g transform="translate(35, 60)">
              <circle cx="0" cy="0" r="28" class="shape-fill" />
              <circle cx="0" cy="0" r="28" class="shape-stroke" fill="none" />
              <!-- Equator line for 3D hint -->
              <ellipse cx="0" cy="0" rx="28" ry="10" class="shape-stroke" fill="none" />
            </g>
            <!-- Cylinder (same radius 28 as sphere) -->
            <g transform="translate(105, 60)">
              <ellipse cx="0" cy="-20" rx="28" ry="10" class="shape-fill" />
              <rect x="-28" y="-20" width="56" height="40" class="shape-fill" />
              <ellipse cx="0" cy="20" rx="28" ry="10" class="shape-fill" />
              <!-- Outline -->
              <line x1="-28" y1="-20" x2="-28" y2="20" class="shape-stroke" />
              <line x1="28" y1="-20" x2="28" y2="20" class="shape-stroke" />
              <ellipse cx="0" cy="-20" rx="28" ry="10" class="shape-stroke" fill="none" />
              <ellipse cx="0" cy="20" rx="28" ry="10" class="shape-stroke" fill="none" />
            </g>
          </g>

          <!-- Arrow B -->
          <g transform="translate(175, 60)">
            <line x1="0" y1="0" x2="40" y2="0" class="arrow-line" marker-end="url(#arrowhead)" />
            <foreignObject x="5" y="-30" width="30" height="25">
              <div class="math-label" v-html="mathB" />
            </foreignObject>
          </g>

          <!-- Clean union result -->
          <g transform="translate(230, 0)">
            <rect x="0" y="0" width="100" height="120" rx="8" class="inner-box" />
            <g transform="translate(50, 60)">
              <!-- Hemisphere bottom (sphere radius 28) -->
              <path d="M -28 0 A 28 28 0 0 0 28 0" class="shape-fill" />
              <!-- Cylinder on top (same radius 28) -->
              <ellipse cx="0" cy="-30" rx="28" ry="10" class="shape-fill" />
              <rect x="-28" y="-30" width="56" height="30" class="shape-fill" />
              <!-- Outline -->
              <line x1="-28" y1="-30" x2="-28" y2="0" class="shape-stroke" />
              <line x1="28" y1="-30" x2="28" y2="0" class="shape-stroke" />
              <ellipse cx="0" cy="-30" rx="28" ry="10" class="shape-stroke" fill="none" />
              <path d="M -28 0 A 28 28 0 0 0 28 0" class="shape-stroke" fill="none" />
            </g>
          </g>

          <!-- Dashed line to graph -->
          <g transform="translate(345, 60)">
            <line x1="0" y1="0" x2="35" y2="0" class="dashed-line" />
            <foreignObject x="-2" y="-25" width="40" height="25">
              <div class="math-label-small" v-html="mathG" />
            </foreignObject>
          </g>

          <!-- Clean intersection graph -->
          <g transform="translate(395, 0)">
            <rect x="0" y="0" width="105" height="120" rx="8" class="inner-box" />
            <g transform="translate(52, 60)">
              <ellipse cx="0" cy="0" rx="40" ry="22" class="graph-stroke" fill="none" />
              <!-- Nodes on ellipse -->
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
    </div>
    <p class="diagram-caption">
      Boolean operation <span v-html="capB" /> commutes with mesh idealization <span v-html="capF" />. <strong>Top:</strong> <span v-html="capB" /> on a mesh with artifacts produces a result with artifacts; the intersection graph <span v-html="capG" /> contains a branching edge. <strong>Bottom:</strong> <span v-html="capB" /> on idealized meshes produces a clean result; <span v-html="capG" /> is a simple closed curve. Apply <span v-html="capF" /> at any point—both paths yield equivalent geometry.
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

.math-label {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
}

.math-label-small {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.9rem;
}

.math-label :deep(.katex),
.math-label-small :deep(.katex) {
  color: inherit;
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

<script setup lang="ts">
</script>

<template>
  <div class="python-query-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <svg viewBox="0 0 500 604" class="w-full">
        <defs>
          <marker id="prim-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
          </marker>
        </defs>

        <!-- Column labels -->
        <text x="145" y="14" class="stage-label" text-anchor="middle">Single</text>
        <text x="355" y="14" class="stage-label" text-anchor="middle">Batch</text>

        <!-- np.array boxes -->
        <g transform="translate(145, 24)">
          <rect x="-75" y="0" width="150" height="44" rx="6" class="stage-box" />
          <text x="0" y="18" class="code-tiny" text-anchor="middle">
            <tspan class="syn-var">np</tspan><tspan>.</tspan><tspan class="syn-type">array</tspan>
          </text>
          <text x="0" y="35" class="desc-text" text-anchor="middle">(shape)</text>
        </g>

        <g transform="translate(355, 24)">
          <rect x="-75" y="0" width="150" height="44" rx="6" class="stage-box" />
          <text x="0" y="18" class="code-tiny" text-anchor="middle">
            <tspan class="syn-var">np</tspan><tspan>.</tspan><tspan class="syn-type">array</tspan>
          </text>
          <text x="0" y="35" class="desc-text" text-anchor="middle">(N, *shape)</text>
        </g>

        <!-- Arrows straight down -->
        <path d="M 145 70 L 145 100" class="arrow-line" marker-end="url(#prim-arrow)" />
        <path d="M 355 70 L 355 100" class="arrow-line" marker-end="url(#prim-arrow)" />

        <!-- Primitive boxes (enriched) -->
        <g transform="translate(145, 108)">
          <rect x="-70" y="0" width="140" height="30" rx="6" class="stage-box stage-enriched" />
          <text x="0" y="20" class="code-tiny" text-anchor="middle">
            <tspan class="syn-type">Primitive</tspan>
          </text>
        </g>

        <g transform="translate(355, 108)">
          <rect x="-70" y="0" width="140" height="30" rx="6" class="stage-box stage-enriched" />
          <text x="0" y="20" class="code-tiny" text-anchor="middle">
            N × <tspan class="syn-type">Primitive</tspan>
          </text>
        </g>

        <!-- Arrows straight down into primitives container -->
        <path d="M 145 140 L 145 170" class="arrow-line" marker-end="url(#prim-arrow)" />
        <path d="M 355 140 L 355 170" class="arrow-line" marker-end="url(#prim-arrow)" />

        <!-- Primitives container (dashed) -->
        <g transform="translate(250, 188)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Primitives</text>
          <rect x="-190" y="0" width="380" height="80" rx="10" class="stage-box stage-container" />

          <!-- Row 1 -->
          <g transform="translate(-170, 10)">
            <rect x="0" y="0" width="80" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="40" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">Point</tspan></text>
          </g>
          <g transform="translate(-82, 10)">
            <rect x="0" y="0" width="80" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="40" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">Line</tspan></text>
          </g>
          <g transform="translate(6, 10)">
            <rect x="0" y="0" width="80" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="40" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">Ray</tspan></text>
          </g>
          <g transform="translate(94, 10)">
            <rect x="0" y="0" width="80" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="40" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">Segment</tspan></text>
          </g>

          <!-- Row 2 -->
          <g transform="translate(-170, 44)">
            <rect x="0" y="0" width="80" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="40" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">Triangle</tspan></text>
          </g>
          <g transform="translate(-82, 44)">
            <rect x="0" y="0" width="80" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="40" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">Polygon</tspan></text>
          </g>
          <g transform="translate(6, 44)">
            <rect x="0" y="0" width="80" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="40" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">Plane</tspan></text>
          </g>
          <g transform="translate(94, 44)">
            <rect x="0" y="0" width="80" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="40" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">AABB</tspan></text>
          </g>
        </g>

        <!-- Arrow down from primitives to forms -->
        <path d="M 250 270 L 250 304" class="arrow-line" marker-end="url(#prim-arrow)" />

        <!-- Forms container (dashed) -->
        <g transform="translate(250, 338)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Forms</text>
          <rect x="-190" y="0" width="380" height="112" rx="10" class="stage-box stage-container" />

          <!-- Inner box covering 3 form types -->
          <g transform="translate(-170, 10)">
            <rect x="0" y="0" width="340" height="40" rx="8" class="stage-box" />
            <g transform="translate(15, 7)">
              <rect x="0" y="0" width="90" height="26" rx="5" class="stage-box stage-enriched" />
              <text x="45" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">Mesh</tspan></text>
            </g>
            <g transform="translate(120, 7)">
              <rect x="0" y="0" width="100" height="26" rx="5" class="stage-box stage-enriched" />
              <text x="50" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">EdgeMesh</tspan></text>
            </g>
            <g transform="translate(235, 7)">
              <rect x="0" y="0" width="90" height="26" rx="5" class="stage-box stage-enriched" />
              <text x="45" y="18" class="code-tiny" text-anchor="middle"><tspan class="syn-type">PointCloud</tspan></text>
            </g>
          </g>

          <!-- Transformation box -->
          <g transform="translate(-100, 64)">
            <rect x="0" y="0" width="200" height="30" rx="6" class="stage-box" />
            <text x="100" y="20" class="code-tiny" text-anchor="middle">
              <tspan class="syn-var">transformation</tspan>
            </text>
          </g>
        </g>

        <!-- Arrow down from forms to cached properties -->
        <path d="M 250 452 L 250 478" class="arrow-line" marker-end="url(#prim-arrow)" />

        <!-- Cached properties container -->
        <g transform="translate(250, 510)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Cached on demand</text>
          <rect x="-190" y="0" width="380" height="80" rx="10" class="stage-box stage-container" />

          <!-- Property boxes -->
          <g transform="translate(-170, 10)">
            <rect x="0" y="0" width="115" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-tiny"><tspan class="syn-var">.normals</tspan></text>
          </g>
          <g transform="translate(-45, 10)">
            <rect x="0" y="0" width="115" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-tiny"><tspan class="syn-var">.vertex_link</tspan></text>
          </g>
          <g transform="translate(80, 10)">
            <rect x="0" y="0" width="100" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-tiny"><tspan class="syn-var">.face_link</tspan></text>
          </g>

          <g transform="translate(-170, 44)">
            <rect x="0" y="0" width="115" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-tiny"><tspan class="syn-var">.point_normals</tspan></text>
          </g>
          <g transform="translate(-45, 44)">
            <rect x="0" y="0" width="115" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-tiny"><tspan class="syn-var">.spatial_tree</tspan></text>
          </g>
          <g transform="translate(80, 44)">
            <rect x="0" y="0" width="100" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="50" y="18" class="ellipsis-text" text-anchor="middle">...</text>
          </g>
        </g>

      </svg>
    </div>
    <p class="diagram-caption">
      NumPy arrays become typed primitives — single or batch. Forms add spatial indexing, on-demand topology and geometry caching, and transformations.
    </p>
  </div>
</template>

<style scoped>
.python-query-diagram {
  color: var(--color-neutral-900);
}

.dark .python-query-diagram {
  color: var(--color-neutral-100);
}

.diagram-box {
  background: rgba(0, 0, 0, 0.02);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  padding: 1rem 0.25rem;
  max-width: 32rem;
  width: 100%;
}

.dark .diagram-box {
  background: rgba(255, 255, 255, 0.03);
  border-color: rgba(255, 255, 255, 0.08);
}

.stage-box {
  fill: rgba(0, 0, 0, 0.04);
  stroke: rgba(0, 0, 0, 0.12);
  stroke-width: 1;
}

.dark .stage-box {
  fill: rgba(255, 255, 255, 0.06);
  stroke: rgba(255, 255, 255, 0.12);
}

.stage-enriched {
  fill: rgba(20, 184, 166, 0.08);
  stroke: rgba(20, 184, 166, 0.3);
}

.dark .stage-enriched {
  fill: rgba(20, 184, 166, 0.12);
  stroke: rgba(20, 184, 166, 0.4);
}

.stage-container {
  fill: rgba(0, 0, 0, 0.02);
  stroke: rgba(0, 0, 0, 0.12);
  stroke-dasharray: 4 2;
}

.dark .stage-container {
  fill: rgba(255, 255, 255, 0.02);
  stroke: rgba(255, 255, 255, 0.12);
}

.stage-label {
  font-size: 11px;
  font-weight: 500;
  fill: currentColor;
  opacity: 0.5;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.ellipsis-text {
  font-size: 11px;
  font-weight: 600;
  fill: currentColor;
  opacity: 0.6;
}

.arrow-line {
  stroke: currentColor;
  stroke-width: 1.5;
  fill: none;
  opacity: 0.4;
}

.code-tiny {
  font-family: 'SF Mono', SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  font-size: 12px;
  fill: currentColor;
}

.desc-text {
  font-family: 'SF Mono', SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  font-size: 10.5px;
  fill: currentColor;
  opacity: 0.4;
}

/* VS Code Light+ theme colors */
.syn-type {
  fill: #267f99;
}

.syn-fn {
  fill: #795E26;
}

.syn-var {
  fill: #001080;
}

/* VS Code Dark+ theme colors */
.dark .syn-type {
  fill: #4EC9B0;
}

.dark .syn-fn {
  fill: #DCDCAA;
}

.dark .syn-var {
  fill: #9CDCFE;
}

.diagram-caption {
  text-align: center;
  font-size: 0.85rem;
  color: var(--color-neutral-500);
  font-style: italic;
  margin-top: 0.75rem;
  max-width: 30rem;
  line-height: 1.5;
}

.dark .diagram-caption {
  color: var(--color-neutral-400);
}
</style>

<script setup lang="ts">
</script>

<template>
  <div class="python-flow-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <svg viewBox="0 0 500 708" class="w-full">
        <!-- Arrowhead marker -->
        <defs>
          <marker id="py-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
          </marker>
        </defs>

        <!-- Stage 1: NumPy Arrays Input - two boxes -->
        <g transform="translate(155, 25)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">faces</text>
          <rect x="-85" y="0" width="170" height="36" rx="6" class="stage-box" />
          <text x="0" y="23" class="code-text" text-anchor="middle">
            <tspan class="syn-var">np</tspan><tspan>.</tspan><tspan class="syn-type">array</tspan>
          </text>
        </g>

        <g transform="translate(345, 25)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">points</text>
          <rect x="-85" y="0" width="170" height="36" rx="6" class="stage-box" />
          <text x="0" y="23" class="code-text" text-anchor="middle">
            <tspan class="syn-var">np</tspan><tspan>.</tspan><tspan class="syn-type">array</tspan>
          </text>
        </g>

        <!-- Arrows converging -->
        <path d="M 185 66 L 220 95" class="arrow-line" marker-end="url(#py-arrow)" />
        <path d="M 315 66 L 280 95" class="arrow-line" marker-end="url(#py-arrow)" />

        <!-- Stage 2: Mesh -->
        <g transform="translate(250, 115)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Form</text>
          <rect x="-130" y="0" width="260" height="36" rx="6" class="stage-box stage-enriched" />
          <text x="0" y="23" class="code-text" text-anchor="middle">
            <tspan class="syn-var">mesh</tspan><tspan> = </tspan><tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-type">Mesh</tspan><tspan>(</tspan><tspan class="syn-var">faces</tspan><tspan>, </tspan><tspan class="syn-var">points</tspan><tspan>)</tspan>
          </text>
        </g>

        <!-- Arrow down -->
        <path d="M 250 156 L 250 185" class="arrow-line" marker-end="url(#py-arrow)" />

        <!-- Stage 3: Properties -->
        <g transform="translate(250, 215)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Properties (cached on demand)</text>
          <!-- Outer dashed container -->
          <rect x="-155" y="0" width="310" height="80" rx="10" class="stage-box stage-container" />

          <!-- Property boxes - row 1 -->
          <g transform="translate(-135, 12)">
            <rect x="0" y="0" width="130" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-tiny">
              <tspan class="syn-var">.normals</tspan>
            </text>
          </g>
          <g transform="translate(5, 12)">
            <rect x="0" y="0" width="130" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-tiny">
              <tspan class="syn-var">.vertex_link</tspan>
            </text>
          </g>

          <!-- Property boxes - row 2 -->
          <g transform="translate(-135, 44)">
            <rect x="0" y="0" width="130" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-tiny">
              <tspan class="syn-var">.face_link</tspan>
            </text>
          </g>
          <g transform="translate(5, 44)">
            <rect x="0" y="0" width="130" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="65" y="18" class="ellipsis-text" text-anchor="middle">...</text>
          </g>
        </g>

        <!-- Arrows branching to two shared views -->
        <path d="M 165 300 L 120 340" class="arrow-line" marker-end="url(#py-arrow)" />
        <path d="M 335 300 L 380 340" class="arrow-line" marker-end="url(#py-arrow)" />

        <!-- Stage 4a: mesh0 shared_view -->
        <g transform="translate(120, 365)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Shared View</text>
          <rect x="-110" y="0" width="220" height="45" rx="6" class="stage-box stage-enriched" />
          <text x="-100" y="17" class="code-tiny">
            <tspan class="syn-var">mesh0</tspan><tspan> = </tspan><tspan class="syn-var">mesh</tspan><tspan>.</tspan><tspan class="syn-fn">shared_view</tspan><tspan>()</tspan>
          </text>
          <text x="-100" y="35" class="code-tiny">
            <tspan class="syn-var">mesh0</tspan><tspan>.</tspan><tspan class="syn-var">transformation</tspan><tspan> = </tspan><tspan class="syn-var">t0</tspan>
          </text>
        </g>

        <!-- Stage 4b: mesh1 shared_view -->
        <g transform="translate(380, 365)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Shared View</text>
          <rect x="-110" y="0" width="220" height="45" rx="6" class="stage-box stage-enriched" />
          <text x="-100" y="17" class="code-tiny">
            <tspan class="syn-var">mesh1</tspan><tspan> = </tspan><tspan class="syn-var">mesh</tspan><tspan>.</tspan><tspan class="syn-fn">shared_view</tspan><tspan>()</tspan>
          </text>
          <text x="-100" y="35" class="code-tiny">
            <tspan class="syn-var">mesh1</tspan><tspan>.</tspan><tspan class="syn-var">transformation</tspan><tspan> = </tspan><tspan class="syn-var">t1</tspan>
          </text>
        </g>

        <!-- Arrows converging to algorithms -->
        <path d="M 165 415 L 210 450" class="arrow-line" marker-end="url(#py-arrow)" />
        <path d="M 335 415 L 290 450" class="arrow-line" marker-end="url(#py-arrow)" />

        <!-- Stage 5: Algorithms -->
        <g transform="translate(250, 475)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Algorithms</text>
          <!-- Outer container -->
          <rect x="-175" y="0" width="350" height="110" rx="10" class="stage-box stage-container" />

          <!-- Inner box: boolean_union -->
          <g transform="translate(-155, 12)">
            <rect x="0" y="0" width="310" height="28" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="19" class="code-small">
              <tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-fn">boolean_union</tspan><tspan>(</tspan><tspan class="syn-var">mesh0</tspan><tspan>, </tspan><tspan class="syn-var">mesh1</tspan><tspan>)</tspan>
            </text>
          </g>

          <!-- Ellipsis box -->
          <g transform="translate(-25, 47)">
            <rect x="0" y="0" width="50" height="16" rx="3" class="stage-box stage-enriched" />
            <text x="25" y="12" class="ellipsis-text" text-anchor="middle">...</text>
          </g>

          <!-- Inner box: intersection_curves -->
          <g transform="translate(-155, 70)">
            <rect x="0" y="0" width="310" height="28" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="19" class="code-small">
              <tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-fn">intersection_curves</tspan><tspan>(</tspan><tspan class="syn-var">mesh0</tspan><tspan>, </tspan><tspan class="syn-var">mesh1</tspan><tspan>)</tspan>
            </text>
          </g>
        </g>

        <!-- Arrow down -->
        <path d="M 250 590 L 250 630" class="arrow-line" marker-end="url(#py-arrow)" />

        <!-- Stage 6: Result -->
        <g transform="translate(250, 655)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Result</text>
          <rect x="-100" y="0" width="200" height="36" rx="6" class="stage-box" />
          <text x="0" y="23" class="code-text" text-anchor="middle">
            <tspan class="syn-var">np</tspan><tspan>.</tspan><tspan class="syn-type">array</tspan><tspan>, ...</tspan>
          </text>
        </g>
      </svg>
    </div>
    <p class="diagram-caption">
      NumPy in, NumPy out. Forms cache properties on demand.
    </p>
  </div>
</template>

<style scoped>
.python-flow-diagram {
  color: var(--color-neutral-900);
}

.dark .python-flow-diagram {
  color: var(--color-neutral-100);
}

.diagram-box {
  background: rgba(0, 0, 0, 0.02);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  padding: 1rem 0.5rem;
  max-width: 30rem;
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

.stage-inner-container {
  fill: rgba(0, 0, 0, 0.02);
  stroke: rgba(0, 0, 0, 0.06);
  stroke-width: 1;
}

.dark .stage-inner-container {
  fill: rgba(255, 255, 255, 0.02);
  stroke: rgba(255, 255, 255, 0.06);
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

.divider-line {
  stroke: currentColor;
  stroke-width: 1;
  opacity: 0.1;
}

.arrow-line {
  stroke: currentColor;
  stroke-width: 1.5;
  fill: none;
  opacity: 0.4;
}

/* Code text styles with monospace font */
.code-text {
  font-family: 'SF Mono', SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  font-size: 13px;
  fill: currentColor;
}

.code-small {
  font-family: 'SF Mono', SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  font-size: 12.5px;
  fill: currentColor;
}

.code-tiny {
  font-family: 'SF Mono', SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  font-size: 12px;
  fill: currentColor;
}

/* VS Code Light+ theme colors */
.syn-kw {
  fill: #0000ff;
}

.syn-ctrl {
  fill: #af00db;
}

.syn-type {
  fill: #267f99;
}

.syn-fn {
  fill: #795E26;
}

.syn-var {
  fill: #001080;
}

.syn-num {
  fill: #098658;
}

.syn-str {
  fill: #a31515;
}

/* VS Code Dark+ theme colors */
.dark .syn-kw {
  fill: #569CD6;
}

.dark .syn-ctrl {
  fill: #C586C0;
}

.dark .syn-type {
  fill: #4EC9B0;
}

.dark .syn-fn {
  fill: #DCDCAA;
}

.dark .syn-var {
  fill: #9CDCFE;
}

.dark .syn-num {
  fill: #B5CEA8;
}

.dark .syn-str {
  fill: #CE9178;
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

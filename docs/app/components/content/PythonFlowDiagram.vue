<script setup lang="ts">
</script>

<template>
  <div class="python-flow-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <svg viewBox="0 0 500 600" class="w-full">
        <defs>
          <marker id="q-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
          </marker>
        </defs>

        <!-- Primitive box at top -->
        <g transform="translate(250, 18)">
          <rect x="-80" y="0" width="160" height="30" rx="6" class="stage-box stage-enriched" />
          <text x="0" y="20" class="code-tiny" text-anchor="middle">
            <tspan class="syn-type">Primitive</tspan>
          </text>
        </g>

        <!-- Arrow down from Primitive to Queries -->
        <path d="M 250 50 L 250 80" class="arrow-line" marker-end="url(#q-arrow)" />

        <!-- Queries container (dashed) -->
        <g transform="translate(250, 112)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Queries</text>
          <rect x="-195" y="0" width="390" height="222" rx="10" class="stage-box stage-container" />

          <!-- Inner box: function names -->
          <g transform="translate(-175, 10)">
            <rect x="0" y="0" width="350" height="72" rx="8" class="stage-box" />

            <!-- Row 1 -->
            <g transform="translate(10, 7)">
              <rect x="0" y="0" width="105" height="26" rx="5" class="stage-box stage-enriched" />
              <text x="8" y="18" class="code-tiny">
                <tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-fn">distance</tspan>
              </text>
            </g>
            <g transform="translate(125, 7)">
              <rect x="0" y="0" width="115" height="26" rx="5" class="stage-box stage-enriched" />
              <text x="8" y="18" class="code-tiny">
                <tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-fn">intersects</tspan>
              </text>
            </g>
            <g transform="translate(250, 7)">
              <rect x="0" y="0" width="90" height="26" rx="5" class="stage-box stage-enriched" />
              <text x="8" y="18" class="code-tiny">
                <tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-fn">ray_cast</tspan>
              </text>
            </g>

            <!-- Row 2 -->
            <g transform="translate(10, 37)">
              <rect x="0" y="0" width="230" height="26" rx="5" class="stage-box stage-enriched" />
              <text x="8" y="18" class="code-tiny">
                <tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-fn">closest_metric_point_pair</tspan>
              </text>
            </g>
            <g transform="translate(250, 37)">
              <rect x="0" y="0" width="90" height="26" rx="5" class="stage-box stage-enriched" />
              <text x="45" y="18" class="ellipsis-text" text-anchor="middle">...</text>
            </g>
          </g>

          <!-- Inner box: broadcasting table -->
          <g transform="translate(-175, 94)">
            <rect x="0" y="0" width="350" height="118" rx="8" class="stage-box" />

            <g transform="translate(0, 20)">
              <text x="15" y="0" class="code-tiny"><tspan class="syn-type">primitive</tspan></text>
              <text x="112" y="1" class="code-tiny" opacity="0.5">×</text>
              <text x="130" y="0" class="code-tiny"><tspan class="syn-type">primitive</tspan></text>
              <text x="240" y="1" class="code-tiny" opacity="0.5">→</text>
              <text x="260" y="0" class="code-tiny"><tspan class="syn-type">float</tspan></text>
            </g>
            <g transform="translate(0, 40)">
              <text x="15" y="0" class="code-tiny"><tspan class="syn-type">batch</tspan>(N)</text>
              <text x="112" y="1" class="code-tiny" opacity="0.5">×</text>
              <text x="130" y="0" class="code-tiny"><tspan class="syn-type">primitive</tspan></text>
              <text x="240" y="1" class="code-tiny" opacity="0.5">→</text>
              <text x="260" y="0" class="code-tiny"><tspan class="syn-type">ndarray</tspan>(N,)</text>
            </g>
            <g transform="translate(0, 60)">
              <text x="15" y="0" class="code-tiny"><tspan class="syn-type">batch</tspan>(N)</text>
              <text x="112" y="1" class="code-tiny" opacity="0.5">×</text>
              <text x="130" y="0" class="code-tiny"><tspan class="syn-type">batch</tspan>(N)</text>
              <text x="240" y="1" class="code-tiny" opacity="0.5">→</text>
              <text x="260" y="0" class="code-tiny"><tspan class="syn-type">ndarray</tspan>(N,)</text>
            </g>
            <g transform="translate(0, 80)">
              <text x="15" y="0" class="code-tiny"><tspan class="syn-type">batch</tspan>(N)</text>
              <text x="112" y="1" class="code-tiny" opacity="0.5">×</text>
              <text x="130" y="0" class="code-tiny"><tspan class="syn-type">form</tspan></text>
              <text x="240" y="1" class="code-tiny" opacity="0.5">→</text>
              <text x="260" y="0" class="code-tiny"><tspan class="syn-type">ndarray</tspan>(N,)</text>
            </g>
            <g transform="translate(0, 100)">
              <text x="15" y="0" class="code-tiny"><tspan class="syn-type">form</tspan></text>
              <text x="112" y="1" class="code-tiny" opacity="0.5">×</text>
              <text x="130" y="0" class="code-tiny"><tspan class="syn-type">form</tspan></text>
              <text x="240" y="1" class="code-tiny" opacity="0.5">→</text>
              <text x="260" y="0" class="code-tiny"><tspan class="syn-type">float</tspan></text>
            </g>
          </g>
        </g>

        <!-- Arrow up from Form to Queries -->
        <path d="M 250 366 L 250 340" class="arrow-line" marker-end="url(#q-arrow)" />

        <!-- Form box (center hub) -->
        <g transform="translate(250, 374)">
          <rect x="-60" y="0" width="120" height="30" rx="6" class="stage-box stage-enriched" />
          <text x="0" y="20" class="code-tiny" text-anchor="middle">
            <tspan class="syn-type">Form</tspan>
          </text>
        </g>

        <!-- Arrow down from Form to Algorithms -->
        <path d="M 250 406 L 250 434" class="arrow-line" marker-end="url(#q-arrow)" />

        <!-- Algorithms container (dashed) -->
        <g transform="translate(250, 462)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Algorithms</text>
          <rect x="-195" y="0" width="390" height="126" rx="10" class="stage-box stage-container" />

          <!-- boolean_union -->
          <g transform="translate(-175, 10)">
            <rect x="0" y="0" width="350" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-small">
              <tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-fn">boolean_union</tspan><tspan>(</tspan><tspan class="syn-var">mesh0</tspan><tspan>, </tspan><tspan class="syn-var">mesh1</tspan><tspan>)</tspan>
            </text>
          </g>

          <!-- fit_icp_alignment -->
          <g transform="translate(-175, 40)">
            <rect x="0" y="0" width="350" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-small">
              <tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-fn">fit_icp_alignment</tspan><tspan>(</tspan><tspan class="syn-var">cloud0</tspan><tspan>, </tspan><tspan class="syn-var">cloud1</tspan><tspan>)</tspan>
            </text>
          </g>

          <!-- Ellipsis -->
          <g transform="translate(-25, 70)">
            <rect x="0" y="0" width="50" height="16" rx="3" class="stage-box stage-enriched" />
            <text x="25" y="12" class="ellipsis-text" text-anchor="middle">...</text>
          </g>

          <!-- decimated -->
          <g transform="translate(-175, 90)">
            <rect x="0" y="0" width="350" height="26" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="18" class="code-small">
              <tspan class="syn-var">tf</tspan><tspan>.</tspan><tspan class="syn-fn">decimated</tspan><tspan>(</tspan><tspan class="syn-var">mesh</tspan><tspan>, </tspan><tspan class="syn-num">0.1</tspan><tspan>)</tspan>
            </text>
          </g>
        </g>

      </svg>
    </div>
    <p class="diagram-caption">
      One API for any combination of primitives and forms. Queries broadcast. Forms bridge spatial queries and mesh algorithms.
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

.dark .syn-num {
  fill: #B5CEA8;
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

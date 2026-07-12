<script setup lang="ts">
</script>

<template>
  <div class="arrangements-pipeline-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <svg viewBox="0 0 540 880" class="w-full">
        <defs>
          <marker id="ap-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
          </marker>
        </defs>

        <!-- Inputs -->
        <g transform="translate(270, 30)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Inputs</text>
          <rect x="-240" y="0" width="480" height="74" rx="10" class="stage-box stage-container" />
          <g transform="translate(-220, 14)">
            <rect x="0" y="0" width="140" height="48" rx="6" class="stage-box stage-enriched" />
            <text x="70" y="20" class="code-tiny" text-anchor="middle"><tspan class="syn-var">straddle</tspan></text>
            <text x="70" y="38" class="code-tiny" text-anchor="middle"><tspan class="syn-type">closed</tspan><tspan opacity="0.55"> · op </tspan><tspan class="syn-num">0</tspan></text>
          </g>
          <g transform="translate(-70, 14)">
            <rect x="0" y="0" width="140" height="48" rx="6" class="stage-box stage-enriched" />
            <text x="70" y="20" class="code-tiny" text-anchor="middle"><tspan class="syn-var">floaters</tspan></text>
            <text x="70" y="38" class="code-tiny" text-anchor="middle"><tspan class="syn-type">closed</tspan><tspan opacity="0.55"> · ops </tspan><tspan class="syn-num">1,2</tspan></text>
          </g>
          <g transform="translate(80, 14)">
            <rect x="0" y="0" width="140" height="48" rx="6" class="stage-box stage-enriched" />
            <text x="70" y="20" class="code-tiny" text-anchor="middle"><tspan class="syn-var">knife</tspan></text>
            <text x="70" y="38" class="code-tiny" text-anchor="middle"><tspan class="syn-type">sheet</tspan><tspan opacity="0.55"> · op </tspan><tspan class="syn-num">3</tspan></text>
          </g>
        </g>

        <path d="M 270 110 L 270 163" class="arrow-line" marker-end="url(#ap-arrow)" />
        <text x="278" y="143" class="step-label">
          <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_csg_graph</tspan><tspan>(</tspan><tspan class="syn-var">forms</tspan><tspan>, </tspan><tspan class="syn-var">sheets</tspan><tspan>)</tspan>
        </text>

        <!-- One build -->
        <g transform="translate(270, 195)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">CSG graph — built once, queried three ways</text>
          <rect x="-240" y="0" width="480" height="52" rx="8" class="stage-box" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">graph</tspan>
            <tspan opacity="0.55"> — arrangement + domain classification</tspan>
          </text>
          <text x="-225" y="40" class="code-tiny" opacity="0.55">every query below reuses this build</text>
        </g>

        <!-- Path 1: boolean mesh -->
        <path d="M 270 253 L 270 306" class="arrow-line" marker-end="url(#ap-arrow)" />
        <text x="278" y="286" class="step-label">
          <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_csg_mesh</tspan><tspan>(</tspan><tspan class="syn-var">graph</tspan><tspan>, </tspan><tspan class="syn-fn">difference</tspan><tspan>(</tspan><tspan class="syn-var">solids</tspan><tspan>, </tspan><tspan class="syn-num">3</tspan><tspan>))</tspan>
        </text>
        <g transform="translate(270, 338)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Path 1 — boolean mesh</text>
          <rect x="-240" y="0" width="480" height="52" rx="8" class="stage-box stage-enriched" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">above_mesh</tspan><tspan opacity="0.55"> / </tspan><tspan class="syn-var">below_mesh</tspan>
            <tspan opacity="0.55"> — one closed, capped mesh per side</tspan>
          </text>
          <text x="-225" y="40" class="code-tiny" opacity="0.55">when the sides are all you need</text>
        </g>

        <!-- Path 2: domains by expression -->
        <path d="M 270 396 L 270 449" class="arrow-line" marker-end="url(#ap-arrow)" />
        <text x="278" y="429" class="step-label">
          <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_csg_domains</tspan><tspan>(</tspan><tspan class="syn-var">graph</tspan><tspan>, </tspan><tspan class="syn-var">e</tspan><tspan>)</tspan>
        </text>
        <g transform="translate(270, 481)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Path 2 — domains by expression</text>
          <rect x="-240" y="0" width="480" height="52" rx="8" class="stage-box stage-enriched" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">cells</tspan><tspan opacity="0.55"> — the selected volumes, individually watertight</tspan>
          </text>
          <text x="-225" y="40" class="code-tiny" opacity="0.55">ids stable across queries on one graph</text>
        </g>

        <!-- Path 3: domains by hand -->
        <path d="M 270 539 L 270 592" class="arrow-line" marker-end="url(#ap-arrow)" />
        <text x="278" y="572" class="step-label">
          <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_csg_domains</tspan><tspan>(</tspan><tspan class="syn-var">graph</tspan><tspan>, </tspan><tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-var">return_index_map</tspan><tspan>)</tspan>
        </text>
        <g transform="translate(270, 624)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Path 3 — domains by hand</text>
          <rect x="-240" y="0" width="480" height="72" rx="8" class="stage-box stage-enriched" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">cells</tspan><tspan opacity="0.55">, </tspan><tspan class="syn-var">ids</tspan><tspan opacity="0.55">, </tspan><tspan class="syn-var">imap</tspan><tspan>.</tspan><tspan class="syn-var">inclusion</tspan>
            <tspan opacity="0.55"> — true iff cell k inside op i</tspan>
          </text>
          <text x="-225" y="42" class="code-tiny" opacity="0.55">every cell classified against every operand</text>
          <text x="-225" y="60" class="code-tiny" opacity="0.55">sheet column = behind the sheet's normal</text>
        </g>

        <path d="M 270 702 L 270 755" class="arrow-line" marker-end="url(#ap-arrow)" />
        <text x="278" y="735" class="step-label">
          <tspan class="syn-var">below</tspan><tspan>[</tspan><tspan class="syn-var">k</tspan><tspan>] = </tspan><tspan class="syn-var">imap</tspan><tspan>.</tspan><tspan class="syn-var">inclusion</tspan><tspan>[</tspan><tspan class="syn-var">k</tspan><tspan>][</tspan><tspan class="syn-num">3</tspan><tspan>]</tspan>
        </text>
        <g transform="translate(270, 787)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Selection is a mask</text>
          <rect x="-240" y="0" width="480" height="52" rx="8" class="stage-box stage-enriched" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">above</tspan><tspan> = !</tspan><tspan class="syn-var">below</tspan>
            <tspan opacity="0.55"> — any boolean combination, no new query</tspan>
          </text>
          <text x="-225" y="40" class="code-tiny" opacity="0.55">same cells as the path-2 expressions, by stable ids</text>
        </g>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.diagram-box {
  width: 100%;
  max-width: 38rem;
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

.arrow-line {
  stroke: currentColor;
  stroke-width: 1.5;
  fill: none;
  opacity: 0.4;
}

.step-label {
  font-family: 'SF Mono', SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  font-size: 11.5px;
  fill: currentColor;
  opacity: 0.85;
}

.code-tiny {
  font-family: 'SF Mono', SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  font-size: 12px;
  fill: currentColor;
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

.syn-ns {
  fill: #267f99;
}

.syn-num {
  fill: #098658;
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

.dark .syn-ns {
  fill: #4EC9B0;
}

.dark .syn-num {
  fill: #B5CEA8;
}
</style>

<script setup lang="ts">
</script>

<template>
  <div class="py-arrangements-pipeline-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <svg viewBox="0 0 540 940" class="w-full">
        <defs>
          <marker id="pyap-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
          </marker>
        </defs>

        <!-- Stage 1: Inputs -->
        <g transform="translate(270, 30)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Inputs</text>
          <rect x="-240" y="0" width="480" height="74" rx="10" class="stage-box stage-container" />

          <g transform="translate(-220, 14)">
            <rect x="0" y="0" width="140" height="48" rx="6" class="stage-box stage-enriched" />
            <text x="70" y="20" class="code-tiny" text-anchor="middle">
              <tspan class="syn-var">cube0</tspan>
            </text>
            <text x="70" y="38" class="code-tiny" text-anchor="middle">
              <tspan class="syn-type">closed</tspan>
              <tspan opacity="0.55"> · tag </tspan>
              <tspan class="syn-num">0</tspan>
            </text>
          </g>

          <g transform="translate(-70, 14)">
            <rect x="0" y="0" width="140" height="48" rx="6" class="stage-box stage-enriched" />
            <text x="70" y="20" class="code-tiny" text-anchor="middle">
              <tspan class="syn-var">cube1</tspan>
            </text>
            <text x="70" y="38" class="code-tiny" text-anchor="middle">
              <tspan class="syn-type">closed</tspan>
              <tspan opacity="0.55"> · tag </tspan>
              <tspan class="syn-num">1</tspan>
            </text>
          </g>

          <g transform="translate(80, 14)">
            <rect x="0" y="0" width="140" height="48" rx="6" class="stage-box stage-enriched" />
            <text x="70" y="20" class="code-tiny" text-anchor="middle">
              <tspan class="syn-var">knife</tspan>
            </text>
            <text x="70" y="38" class="code-tiny" text-anchor="middle">
              <tspan class="syn-type">open plane</tspan>
              <tspan opacity="0.55"> · tag </tspan>
              <tspan class="syn-num">2</tspan>
            </text>
          </g>
        </g>

        <!-- Arrow 1 -->
        <path d="M 270 110 L 270 163" class="arrow-line" marker-end="url(#pyap-arrow)" />
        <text x="278" y="143" class="step-label">
          <tspan class="syn-ns">tf</tspan><tspan>.</tspan><tspan class="syn-fn">mesh_arrangements</tspan>
        </text>

        <!-- Stage 2: Arrangement -->
        <g transform="translate(270, 195)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Arrangement</text>
          <rect x="-240" y="0" width="480" height="62" rx="8" class="stage-box" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">arr</tspan>
            <tspan opacity="0.55"> — merged mesh, every face split at every intersection</tspan>
          </text>
          <text x="-225" y="42" class="code-tiny">
            <tspan class="syn-var">tag_labels</tspan><tspan opacity="0.55">[f] · </tspan><tspan class="syn-var">face_labels</tspan><tspan opacity="0.55">[f]</tspan>
          </text>
        </g>

        <!-- Arrow 2 -->
        <path d="M 270 263 L 270 316" class="arrow-line" marker-end="url(#pyap-arrow)" />
        <text x="278" y="296" class="step-label">
          <tspan class="syn-ns">tf</tspan><tspan>.</tspan><tspan class="syn-fn">cleaned</tspan>
          <tspan opacity="0.55">  +  </tspan>
          <tspan class="syn-var">tag_labels</tspan><tspan>[</tspan><tspan class="syn-var">kept_faces</tspan><tspan>]</tspan>
        </text>

        <!-- Stage 3: Cleaned -->
        <g transform="translate(270, 348)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Cleaned + reindexed tags</text>
          <rect x="-240" y="0" width="480" height="62" rx="8" class="stage-box" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">arr</tspan>
            <tspan opacity="0.55"> — merged vertices, dropped degenerate &amp; duplicate faces</tspan>
          </text>
          <text x="-225" y="42" class="code-tiny">
            <tspan class="syn-var">tag_labels</tspan><tspan opacity="0.55"> aligned to surviving faces via </tspan><tspan class="syn-var">kept_faces</tspan>
          </text>
        </g>

        <!-- Arrow 3 -->
        <path d="M 270 416 L 270 469" class="arrow-line" marker-end="url(#pyap-arrow)" />
        <text x="278" y="449" class="step-label">
          <tspan class="syn-ns">tf</tspan><tspan>.</tspan><tspan class="syn-fn">domain_labels</tspan>
        </text>

        <!-- Stage 4: Domain labels -->
        <g transform="translate(270, 501)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Domain labels</text>
          <rect x="-240" y="0" width="480" height="100" rx="8" class="stage-box" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">labels_2d</tspan><tspan>[</tspan><tspan class="syn-var">f</tspan><tspan>, </tspan><tspan class="syn-num">0</tspan><tspan>]</tspan>
            <tspan opacity="0.55"> — domain containing f with </tspan>
            <tspan class="syn-type">reversed winding</tspan>
          </text>
          <text x="-225" y="42" class="code-tiny">
            <tspan class="syn-var">labels_2d</tspan><tspan>[</tspan><tspan class="syn-var">f</tspan><tspan>, </tspan><tspan class="syn-num">1</tspan><tspan>]</tspan>
            <tspan opacity="0.55"> — domain containing f with </tspan>
            <tspan class="syn-type">forward winding</tspan>
          </text>
          <text x="-225" y="62" class="code-tiny">
            <tspan class="syn-var">n_domains</tspan>
            <tspan opacity="0.55"> — count of bounded interior regions</tspan>
          </text>
          <text x="-225" y="82" class="code-tiny" opacity="0.55">
            flags: <tspan class="syn-var">ignore_open_fragments</tspan><tspan>=</tspan><tspan class="syn-num">True</tspan>, <tspan class="syn-var">exclude_outer_shell</tspan><tspan>=</tspan><tspan class="syn-num">True</tspan>
          </text>
        </g>

        <!-- Arrow 4 -->
        <path d="M 270 607 L 270 660" class="arrow-line" marker-end="url(#pyap-arrow)" />
        <text x="278" y="640" class="step-label">
          <tspan class="syn-ns">tf</tspan><tspan>.</tspan><tspan class="syn-fn">split_into_domains</tspan>
        </text>

        <!-- Stage 5: Volumes -->
        <g transform="translate(270, 692)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Volumes</text>
          <rect x="-240" y="0" width="480" height="80" rx="8" class="stage-box stage-enriched" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">volumes</tspan><tspan>[</tspan><tspan class="syn-var">i</tspan><tspan>]</tspan>
            <tspan opacity="0.55"> — closed · manifold · outward-oriented submesh</tspan>
          </text>
          <text x="-225" y="42" class="code-tiny">
            <tspan class="syn-var">comp_labels</tspan><tspan>[</tspan><tspan class="syn-var">i</tspan><tspan>]</tspan>
            <tspan opacity="0.55"> — domain id of </tspan>
            <tspan class="syn-var">volumes</tspan><tspan>[</tspan><tspan class="syn-var">i</tspan><tspan>]</tspan>
          </text>
          <text x="-225" y="62" class="code-tiny" opacity="0.55">
            side-0 emissions reverse the winding · side-1 emissions keep it
          </text>
        </g>

        <!-- Arrow 5 -->
        <path d="M 270 778 L 270 831" class="arrow-line" marker-end="url(#pyap-arrow)" />
        <text x="278" y="811" class="step-label">
          <tspan class="syn-fn">np.unique</tspan>(<tspan class="syn-var">labels_2d</tspan>[<tspan class="syn-var">knife</tspan>, <tspan class="syn-num">0</tspan><tspan>/</tspan><tspan class="syn-num">1</tspan>])
        </text>

        <!-- Stage 6: Side selection -->
        <g transform="translate(270, 863)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Signed side of the knife</text>
          <rect x="-240" y="0" width="480" height="62" rx="8" class="stage-box stage-enriched" />
          <text x="-225" y="22" class="code-tiny">
            <tspan class="syn-var">above_ids</tspan>
            <tspan opacity="0.55"> — domains receiving the knife with reversed winding</tspan>
          </text>
          <text x="-225" y="42" class="code-tiny">
            <tspan class="syn-var">below_ids</tspan>
            <tspan opacity="0.55"> — domains receiving the knife with forward winding</tspan>
          </text>
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

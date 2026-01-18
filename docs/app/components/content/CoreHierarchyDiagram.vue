<script setup lang="ts">
</script>

<template>
  <div class="core-hierarchy-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <svg viewBox="0 0 580 1036" class="w-full">
        <!-- Arrowhead marker -->
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
          </marker>
        </defs>

        <!-- Stage 1: Raw Data - two boxes -->
        <g transform="translate(160, 25)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Your Data</text>
          <rect x="-90" y="0" width="180" height="36" rx="6" class="stage-box" />
          <text x="0" y="23" class="code-text" text-anchor="middle">
            <tspan class="syn-ns">std</tspan><tspan>::</tspan><tspan class="syn-type">vector</tspan><tspan>&lt;</tspan><tspan class="syn-kw">int</tspan><tspan>&gt;</tspan>
          </text>
        </g>

        <g transform="translate(420, 25)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Your Data</text>
          <rect x="-90" y="0" width="180" height="36" rx="6" class="stage-box" />
          <text x="0" y="23" class="code-text" text-anchor="middle">
            <tspan class="syn-ns">std</tspan><tspan>::</tspan><tspan class="syn-type">vector</tspan><tspan>&lt;</tspan><tspan class="syn-kw">float</tspan><tspan>&gt;</tspan>
          </text>
        </g>

        <!-- Arrows down -->
        <path d="M 160 66 L 160 95" class="arrow-line" marker-end="url(#arrow)" />
        <path d="M 420 66 L 420 95" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 2: Two boxes side by side -->
        <!-- Faces box -->
        <g transform="translate(160, 115)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Faces Range</text>
          <rect x="-120" y="0" width="240" height="75" rx="6" class="stage-box stage-enriched" />
          <text x="-110" y="18" class="code-text">
            <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_faces</tspan><tspan>&lt;</tspan><tspan class="syn-num">3</tspan><tspan>&gt;(</tspan><tspan class="syn-var">ids</tspan><tspan>)</tspan>
          </text>
          <line x1="-110" y1="26" x2="110" y2="26" class="divider-line" />
          <text x="-110" y="42" class="code-small">
            <tspan class="syn-kw">auto</tspan><tspan> [</tspan><tspan class="syn-var">a</tspan><tspan>,</tspan><tspan class="syn-var">b</tspan><tspan>,</tspan><tspan class="syn-var">c</tspan><tspan>] = </tspan><tspan class="syn-var">faces</tspan><tspan>[</tspan><tspan class="syn-var">i</tspan><tspan>];</tspan>
          </text>
          <text x="-110" y="56" class="code-small">
            <tspan class="syn-ctrl">for</tspan><tspan> (</tspan><tspan class="syn-kw">auto</tspan><tspan class="syn-var"> f</tspan><tspan> : </tspan><tspan class="syn-var">faces</tspan><tspan>) { }</tspan>
          </text>
        </g>

        <!-- Points box -->
        <g transform="translate(420, 115)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Points Range</text>
          <rect x="-120" y="0" width="240" height="75" rx="6" class="stage-box stage-enriched" />
          <text x="-110" y="18" class="code-text">
            <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_points</tspan><tspan>&lt;</tspan><tspan class="syn-num">3</tspan><tspan>&gt;(</tspan><tspan class="syn-var">coords</tspan><tspan>)</tspan>
          </text>
          <line x1="-110" y1="26" x2="110" y2="26" class="divider-line" />
          <text x="-110" y="42" class="code-small">
            <tspan class="syn-kw">auto</tspan><tspan class="syn-var"> v</tspan><tspan> = </tspan><tspan class="syn-var">pt</tspan><tspan> + </tspan><tspan class="syn-var">vec</tspan><tspan>;</tspan>
          </text>
          <text x="-110" y="56" class="code-small">
            <tspan class="syn-kw">auto</tspan><tspan class="syn-var"> d</tspan><tspan> = </tspan><tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">dot</tspan><tspan>(</tspan><tspan class="syn-var">a</tspan><tspan>, </tspan><tspan class="syn-var">b</tspan><tspan>);</tspan>
          </text>
        </g>

        <!-- Arrows converging to polygons -->
        <path d="M 220 195 L 255 240" class="arrow-line" marker-end="url(#arrow)" />
        <path d="M 360 195 L 325 240" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 3: Polygons -->
        <g transform="translate(290, 265)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Composed Range</text>
          <rect x="-195" y="0" width="390" height="90" rx="6" class="stage-box stage-enriched" />
          <text x="-185" y="18" class="code-text">
            <tspan class="syn-var">polygons</tspan><tspan> = </tspan><tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_polygons</tspan><tspan>(</tspan><tspan class="syn-var">faces</tspan><tspan>, </tspan><tspan class="syn-var">points</tspan><tspan>)</tspan>
          </text>
          <line x1="-185" y1="26" x2="185" y2="26" class="divider-line" />
          <text x="-185" y="44" class="code-small">
            <tspan class="syn-kw">auto</tspan><tspan class="syn-var"> poly</tspan><tspan> = </tspan><tspan class="syn-var">polygons</tspan><tspan>.</tspan><tspan class="syn-fn">front</tspan><tspan>();</tspan>
          </text>
          <text x="-185" y="58" class="code-small">
            <tspan class="syn-kw">auto</tspan><tspan> [</tspan><tspan class="syn-var">p0</tspan><tspan>, </tspan><tspan class="syn-var">p1</tspan><tspan>, </tspan><tspan class="syn-var">p2</tspan><tspan>] = </tspan><tspan class="syn-var">poly</tspan><tspan>;</tspan>
          </text>
          <text x="-185" y="72" class="code-small">
            <tspan class="syn-kw">auto</tspan><tspan> [</tspan><tspan class="syn-var">id0</tspan><tspan>, </tspan><tspan class="syn-var">id1</tspan><tspan>, </tspan><tspan class="syn-var">id2</tspan><tspan>] = </tspan><tspan class="syn-var">poly</tspan><tspan>.</tspan><tspan class="syn-fn">indices</tspan><tspan>();</tspan>
          </text>
        </g>

        <!-- Arrow down to big container -->
        <path d="M 290 360 L 290 401" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 4: Big container for Optional Structures + Attach Policies -->
        <g transform="translate(290, 431)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Optional Structures</text>
          <!-- Outer dashed container -->
          <rect x="-240" y="0" width="480" height="280" rx="10" class="stage-box stage-container" />

          <!-- Precompute sub-section -->
          <text x="0" y="18" class="stage-label" text-anchor="middle">Precompute</text>
          <rect x="-220" y="28" width="440" height="150" rx="8" class="stage-box stage-inner-container" />

          <!-- Inner box: tree -->
          <g transform="translate(-202, 38)">
            <rect x="0" y="0" width="405" height="32" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="21" class="code-tiny">
              <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-type">aabb_tree</tspan><tspan>&lt;</tspan><tspan class="syn-kw">int</tspan><tspan>, </tspan><tspan class="syn-kw">float</tspan><tspan>, </tspan><tspan class="syn-num">3</tspan><tspan>&gt; </tspan><tspan class="syn-var">tree</tspan><tspan>{</tspan><tspan class="syn-var">polygons</tspan><tspan>};</tspan>
            </text>
          </g>

          <!-- Inner box: face_membership -->
          <g transform="translate(-202, 78)">
            <rect x="0" y="0" width="405" height="32" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="21" class="code-tiny">
              <tspan class="syn-kw">auto</tspan><tspan class="syn-var"> fm</tspan><tspan> = </tspan><tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_face_membership</tspan><tspan>(</tspan><tspan class="syn-var">polygons</tspan><tspan>);</tspan>
            </text>
          </g>

          <!-- Ellipsis box -->
          <g transform="translate(-25, 116)">
            <rect x="0" y="0" width="50" height="16" rx="3" class="stage-box stage-enriched" />
            <text x="25" y="12" class="ellipsis-text" text-anchor="middle">...</text>
          </g>

          <!-- Inner box: manifold_edge_link -->
          <g transform="translate(-202, 138)">
            <rect x="0" y="0" width="405" height="32" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="21" class="code-tiny">
              <tspan class="syn-kw">auto</tspan><tspan class="syn-var"> mel</tspan><tspan> = </tspan><tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_manifold_edge_link</tspan><tspan>(</tspan><tspan class="syn-var">polygons</tspan><tspan>);</tspan>
            </text>
          </g>

          <!-- Arrow from precompute to attach policies -->
          <path d="M 0 183 L 0 210" class="arrow-line" marker-end="url(#arrow)" />

          <!-- Attach Policies sub-section -->
          <text x="0" y="228" class="stage-label" text-anchor="middle">Attach Policies</text>
          <rect x="-220" y="235" width="440" height="28" rx="6" class="stage-box stage-enriched" />
          <text x="-210" y="254" class="code-small">
            <tspan class="syn-var">tagged</tspan><tspan> = </tspan><tspan class="syn-var">polygons</tspan><tspan> | </tspan><tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">tag</tspan><tspan>(</tspan><tspan class="syn-var">tree</tspan><tspan>) | ... | </tspan><tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">tag</tspan><tspan>(</tspan><tspan class="syn-var">mel</tspan><tspan>)</tspan>
          </text>
        </g>

        <!-- Arrows branching to transforms from the big container -->
        <path d="M 200 716 L 145 756" class="arrow-line" marker-end="url(#arrow)" />
        <path d="M 380 716 L 435 756" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 5a: Form0 -->
        <g transform="translate(145, 781)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">+ Transform</text>
          <rect x="-130" y="0" width="260" height="28" rx="6" class="stage-box stage-enriched" />
          <text x="-120" y="19" class="code-small">
            <tspan class="syn-var">form0</tspan><tspan> = </tspan><tspan class="syn-var">tagged</tspan><tspan> | </tspan><tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">tag</tspan><tspan>(</tspan><tspan class="syn-var">tr0</tspan><tspan>)</tspan>
          </text>
        </g>

        <!-- Stage 5b: Form1 -->
        <g transform="translate(435, 781)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">+ Transform</text>
          <rect x="-130" y="0" width="260" height="28" rx="6" class="stage-box stage-enriched" />
          <text x="-120" y="19" class="code-small">
            <tspan class="syn-var">form1</tspan><tspan> = </tspan><tspan class="syn-var">tagged</tspan><tspan> | </tspan><tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">tag</tspan><tspan>(</tspan><tspan class="syn-var">tr1</tspan><tspan>)</tspan>
          </text>
        </g>

        <!-- Arrows converging to operations -->
        <path d="M 205 814 L 250 849" class="arrow-line" marker-end="url(#arrow)" />
        <path d="M 375 814 L 330 849" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 6: Algorithms between forms -->
        <g transform="translate(290, 874)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Algorithms</text>
          <!-- Outer container -->
          <rect x="-175" y="0" width="350" height="145" rx="10" class="stage-box stage-container" />

          <!-- Inner box: make_boolean -->
          <g transform="translate(-155, 12)">
            <rect x="0" y="0" width="310" height="28" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="19" class="code-small">
              <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">make_boolean</tspan><tspan>(</tspan><tspan class="syn-var">form0</tspan><tspan>, </tspan><tspan class="syn-var">form1</tspan><tspan>, </tspan><tspan class="syn-var">op</tspan><tspan>);</tspan>
            </text>
          </g>

          <!-- Inner box: intersects -->
          <g transform="translate(-155, 47)">
            <rect x="0" y="0" width="310" height="28" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="19" class="code-small">
              <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">intersects</tspan><tspan>(</tspan><tspan class="syn-var">form0</tspan><tspan>, </tspan><tspan class="syn-var">form1</tspan><tspan>);</tspan>
            </text>
          </g>

          <!-- Ellipsis box -->
          <g transform="translate(-25, 82)">
            <rect x="0" y="0" width="50" height="16" rx="3" class="stage-box stage-enriched" />
            <text x="25" y="12" class="ellipsis-text" text-anchor="middle">...</text>
          </g>

          <!-- Inner box: distance2 -->
          <g transform="translate(-155, 105)">
            <rect x="0" y="0" width="310" height="28" rx="5" class="stage-box stage-enriched" />
            <text x="10" y="19" class="code-small">
              <tspan class="syn-ns">tf</tspan><tspan>::</tspan><tspan class="syn-fn">distance2</tspan><tspan>(</tspan><tspan class="syn-var">form0</tspan><tspan>, </tspan><tspan class="syn-var">form1</tspan><tspan>);</tspan>
            </text>
          </g>
        </g>
      </svg>
    </div>
    <p class="diagram-caption">
      You choose the complexity; trueform adapts.
    </p>
  </div>
</template>

<style scoped>
.core-hierarchy-diagram {
  color: var(--color-neutral-900);
}

.dark .core-hierarchy-diagram {
  color: var(--color-neutral-100);
}

.diagram-box {
  background: rgba(0, 0, 0, 0.02);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  padding: 1rem 0.5rem;
  max-width: 38rem;
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

.syn-ns {
  fill: #267f99;
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

.dark .syn-ns {
  fill: #4EC9B0;
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
  max-width: 38rem;
  line-height: 1.5;
}

.dark .diagram-caption {
  color: var(--color-neutral-400);
}
</style>

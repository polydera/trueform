<script setup lang="ts">
import { codeToHtml } from 'shiki'

// Pre-highlight code snippets for both themes
const highlightedDark = ref<Record<string, string>>({})
const highlightedLight = ref<Record<string, string>>({})

const colorMode = useColorMode()

const codeSnippets = {
  vectorFloat: 'std::vector<float>',
  vectorInt: 'std::vector<int>',
  points: 'points = tf::make_points<3>(coords)',
  pointsFeature1: 'auto v = pt + vec;',
  pointsFeature2: 'auto d = tf::dot(a, b);',
  pointsFeature3: 'auto c = tf::cross(a, b);',
  faces: 'faces = tf::make_blocked_range<3>(ids)',
  facesFeature1: 'auto [a,b,c] = faces[i];',
  facesFeature2: 'for (auto f : faces) { }',
  polygons: 'polygons = tf::make_polygons(faces, points)',
  polygonsFeature1: 'auto poly = polygons.front();',
  polygonsFeature2: 'auto [p0, p1, p2] = poly;',
  polygonsFeature3: 'auto a = tf::area(poly);',
  polygonsFeature4: 'auto n = tf::make_normal(poly);',
  tree: 'tf::aabb_tree<int, float, 3> tree{polygons};',
  fm: 'auto fm = tf::make_face_membership(polygons);',
  mel: 'auto mel = tf::make_manifold_edge_link(polygons);',
  tagged: 'tagged = polygons | tf::tag(tree) | ... | tf::tag(mel)',
  form0: 'form0 = tagged | tf::tag(transform0)',
  form1: 'form1 = tagged | tf::tag(transform1)',
  op1: 'tf::make_boolean(form0, form1, op);',
  op2: 'tf::distance2(form0, form1);',
  op3: 'tf::intersects(form0, form1);',
}

const highlighted = computed(() => {
  return colorMode.value === 'dark' ? highlightedDark.value : highlightedLight.value
})

onMounted(async () => {
  const darkResults: Record<string, string> = {}
  const lightResults: Record<string, string> = {}
  for (const [key, code] of Object.entries(codeSnippets)) {
    darkResults[key] = await codeToHtml(code, {
      lang: 'cpp',
      theme: 'github-dark-default',
    })
    lightResults[key] = await codeToHtml(code, {
      lang: 'cpp',
      theme: 'github-light-default',
    })
  }
  highlightedDark.value = darkResults
  highlightedLight.value = lightResults
})
</script>

<template>
  <div class="core-hierarchy-diagram my-8 flex flex-col items-center">
    <div class="diagram-box">
      <svg viewBox="0 0 580 976" class="w-full">
        <!-- Arrowhead marker -->
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="currentColor" />
          </marker>
        </defs>

        <!-- Stage 1: Raw Data - two boxes -->
        <g transform="translate(145, 25)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Your Data</text>
          <rect x="-90" y="0" width="180" height="36" rx="6" class="stage-box" />
          <foreignObject x="-85" y="4" width="170" height="28">
            <div class="code-line-wrapper">
              <div v-if="highlighted.vectorFloat" class="shiki-wrapper shiki-line" v-html="highlighted.vectorFloat" />
              <code v-else class="code-line">std::vector&lt;float&gt;</code>
            </div>
          </foreignObject>
        </g>

        <g transform="translate(435, 25)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Your Data</text>
          <rect x="-90" y="0" width="180" height="36" rx="6" class="stage-box" />
          <foreignObject x="-85" y="4" width="170" height="28">
            <div class="code-line-wrapper">
              <div v-if="highlighted.vectorInt" class="shiki-wrapper shiki-line" v-html="highlighted.vectorInt" />
              <code v-else class="code-line">std::vector&lt;int&gt;</code>
            </div>
          </foreignObject>
        </g>

        <!-- Arrows down -->
        <path d="M 145 66 L 145 95" class="arrow-line" marker-end="url(#arrow)" />
        <path d="M 435 66 L 435 95" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 2: Two boxes side by side -->
        <!-- Points box -->
        <g transform="translate(145, 115)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Primitive Range</text>
          <rect x="-138" y="0" width="276" height="75" rx="6" class="stage-box stage-enriched" />
          <foreignObject x="-133" y="4" width="266" height="67">
            <div class="card-content">
              <div v-if="highlighted.points" class="shiki-wrapper" v-html="highlighted.points" />
              <code v-else class="code-create"><span class="hl">points</span> = tf::make_points&lt;3&gt;(coords)</code>
              <div class="card-divider"></div>
              <div v-if="highlighted.pointsFeature1" class="shiki-wrapper shiki-small" v-html="highlighted.pointsFeature1" />
              <code v-else class="code-feature">auto v = pt + vec;</code>
              <div v-if="highlighted.pointsFeature2" class="shiki-wrapper shiki-small" v-html="highlighted.pointsFeature2" />
              <code v-else class="code-feature">auto d = tf::dot(a, b);</code>
            </div>
          </foreignObject>
        </g>

        <!-- Faces box -->
        <g transform="translate(435, 115)">
          <text x="0" y="-8" class="stage-label" text-anchor="middle">Index Range</text>
          <rect x="-138" y="0" width="276" height="75" rx="6" class="stage-box stage-enriched" />
          <foreignObject x="-133" y="4" width="266" height="67">
            <div class="card-content">
              <div v-if="highlighted.faces" class="shiki-wrapper" v-html="highlighted.faces" />
              <code v-else class="code-create"><span class="hl">faces</span> = tf::make_blocked_range&lt;3&gt;(ids)</code>
              <div class="card-divider"></div>
              <div v-if="highlighted.facesFeature1" class="shiki-wrapper shiki-small" v-html="highlighted.facesFeature1" />
              <code v-else class="code-feature">auto [a,b,c] = faces[i];</code>
              <div v-if="highlighted.facesFeature2" class="shiki-wrapper shiki-small" v-html="highlighted.facesFeature2" />
              <code v-else class="code-feature">for (auto f : faces) { }</code>
            </div>
          </foreignObject>
        </g>

        <!-- Arrows converging to polygons -->
        <path d="M 205 195 L 250 240" class="arrow-line" marker-end="url(#arrow)" />
        <path d="M 375 195 L 330 240" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 3: Polygons -->
        <g transform="translate(290, 265)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Composed Range</text>
          <rect x="-165" y="0" width="330" height="115" rx="6" class="stage-box stage-enriched" />
          <foreignObject x="-160" y="4" width="320" height="107">
            <div class="card-content">
              <div v-if="highlighted.polygons" class="shiki-wrapper" v-html="highlighted.polygons" />
              <code v-else class="code-create"><span class="hl">polygons</span> = tf::make_polygons(faces, points)</code>
              <div class="card-divider"></div>
              <div v-if="highlighted.polygonsFeature1" class="shiki-wrapper shiki-small" v-html="highlighted.polygonsFeature1" />
              <code v-else class="code-feature">auto poly = polygons.front();</code>
              <div v-if="highlighted.polygonsFeature2" class="shiki-wrapper shiki-small" v-html="highlighted.polygonsFeature2" />
              <code v-else class="code-feature">auto [p0, p1, p2] = poly;</code>
              <div v-if="highlighted.polygonsFeature3" class="shiki-wrapper shiki-small" v-html="highlighted.polygonsFeature3" />
              <code v-else class="code-feature">auto a = tf::area(poly);</code>
              <div v-if="highlighted.polygonsFeature4" class="shiki-wrapper shiki-small" v-html="highlighted.polygonsFeature4" />
              <code v-else class="code-feature">auto n = tf::make_normal(poly);</code>
            </div>
          </foreignObject>
        </g>

        <!-- Arrow down to big container -->
        <path d="M 290 385 L 290 415" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 4: Big container for Optional Structures + Attach Policies -->
        <g transform="translate(290, 445)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Optional Structures</text>
          <!-- Outer dashed container -->
          <rect x="-240" y="0" width="480" height="280" rx="10" class="stage-box stage-container" />

          <!-- Precompute sub-section -->
          <text x="0" y="18" class="stage-label" text-anchor="middle">Precompute</text>
          <rect x="-220" y="28" width="440" height="150" rx="8" class="stage-box stage-inner-container" />

          <!-- Inner box: tree -->
          <g transform="translate(-202, 38)">
            <rect x="0" y="0" width="405" height="32" rx="5" class="stage-box stage-enriched" />
            <foreignObject x="6" y="8" width="393" height="20">
              <div class="card-content-inner">
                <div v-if="highlighted.tree" class="shiki-wrapper shiki-tiny" v-html="highlighted.tree" />
                <code v-else class="code-tiny">tf::aabb_tree&lt;int, float, 3&gt; tree{polygons};</code>
              </div>
            </foreignObject>
          </g>

          <!-- Inner box: face_membership -->
          <g transform="translate(-202, 78)">
            <rect x="0" y="0" width="405" height="32" rx="5" class="stage-box stage-enriched" />
            <foreignObject x="6" y="8" width="393" height="20">
              <div class="card-content-inner">
                <div v-if="highlighted.fm" class="shiki-wrapper shiki-tiny" v-html="highlighted.fm" />
                <code v-else class="code-tiny">auto fm = tf::make_face_membership(polygons);</code>
              </div>
            </foreignObject>
          </g>

          <!-- Ellipsis box -->
          <g transform="translate(-25, 116)">
            <rect x="0" y="0" width="50" height="16" rx="3" class="stage-box stage-enriched" />
            <text x="25" y="12" class="ellipsis-text" text-anchor="middle">...</text>
          </g>

          <!-- Inner box: manifold_edge_link -->
          <g transform="translate(-202, 138)">
            <rect x="0" y="0" width="405" height="32" rx="5" class="stage-box stage-enriched" />
            <foreignObject x="6" y="8" width="393" height="20">
              <div class="card-content-inner">
                <div v-if="highlighted.mel" class="shiki-wrapper shiki-tiny" v-html="highlighted.mel" />
                <code v-else class="code-tiny">auto mel = tf::make_manifold_edge_link(polygons);</code>
              </div>
            </foreignObject>
          </g>

          <!-- Arrow from precompute to attach policies -->
          <path d="M 0 183 L 0 210" class="arrow-line" marker-end="url(#arrow)" />

          <!-- Attach Policies sub-section -->
          <text x="0" y="228" class="stage-label" text-anchor="middle">Attach Policies</text>
          <rect x="-220" y="235" width="440" height="28" rx="6" class="stage-box stage-enriched" />
          <foreignObject x="-215" y="241" width="430" height="20">
            <div class="card-content">
              <div v-if="highlighted.tagged" class="shiki-wrapper shiki-small" v-html="highlighted.tagged" />
              <code v-else class="code-create"><span class="hl">tagged</span> = polygons | tf::tag(tree) | ... | tf::tag(mel)</code>
            </div>
          </foreignObject>
        </g>

        <!-- Arrows branching to transforms from the big container -->
        <path d="M 200 730 L 145 770" class="arrow-line" marker-end="url(#arrow)" />
        <path d="M 380 730 L 435 770" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 5a: Form0 -->
        <g transform="translate(145, 795)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">+ Transform</text>
          <rect x="-130" y="0" width="260" height="28" rx="6" class="stage-box stage-enriched" />
          <foreignObject x="-125" y="6" width="250" height="20">
            <div class="card-content">
              <div v-if="highlighted.form0" class="shiki-wrapper shiki-small" v-html="highlighted.form0" />
              <code v-else class="code-create"><span class="hl">form0</span> = tagged | tf::tag(transform0)</code>
            </div>
          </foreignObject>
        </g>

        <!-- Stage 5b: Form1 -->
        <g transform="translate(435, 795)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">+ Transform</text>
          <rect x="-130" y="0" width="260" height="28" rx="6" class="stage-box stage-enriched" />
          <foreignObject x="-125" y="6" width="250" height="20">
            <div class="card-content">
              <div v-if="highlighted.form1" class="shiki-wrapper shiki-small" v-html="highlighted.form1" />
              <code v-else class="code-create"><span class="hl">form1</span> = tagged | tf::tag(transform1)</code>
            </div>
          </foreignObject>
        </g>

        <!-- Arrows converging to operations -->
        <path d="M 205 828 L 250 863" class="arrow-line" marker-end="url(#arrow)" />
        <path d="M 375 828 L 330 863" class="arrow-line" marker-end="url(#arrow)" />

        <!-- Stage 6: Algorithms between forms -->
        <g transform="translate(290, 888)">
          <text x="0" y="-10" class="stage-label" text-anchor="middle">Algorithms</text>
          <rect x="-124" y="0" width="248" height="68" rx="6" class="stage-box stage-enriched" />
          <foreignObject x="-119" y="8" width="238" height="56">
            <div class="card-content">
              <div v-if="highlighted.op1" class="shiki-wrapper shiki-small" v-html="highlighted.op1" />
              <code v-else class="code-feature">tf::make_boolean(form0, form1, op);</code>
              <div v-if="highlighted.op2" class="shiki-wrapper shiki-small" v-html="highlighted.op2" />
              <code v-else class="code-feature">tf::distance2(form0, form1);</code>
              <div v-if="highlighted.op3" class="shiki-wrapper shiki-small" v-html="highlighted.op3" />
              <code v-else class="code-feature">tf::intersects(form0, form1);</code>
            </div>
          </foreignObject>
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
  font-size: 10px;
  font-weight: 500;
  fill: currentColor;
  opacity: 0.5;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.ellipsis-text {
  font-size: 10px;
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

.card-content {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.card-divider {
  height: 1px;
  background: currentColor;
  opacity: 0.1;
  margin: 4px 0;
}

.code-line-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.code-line {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  color: inherit;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.shiki-line {
  font-size: 12px;
}

.code-create {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11.5px;
  color: inherit;
  opacity: 0.9;
}

.code-feature {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
  color: inherit;
  opacity: 0.7;
}

.hl {
  color: rgb(20, 184, 166);
}

.shiki-wrapper {
  font-size: 11.5px;
  line-height: 1.4;
}

.shiki-wrapper :deep(pre) {
  margin: 0;
  padding: 0;
  background: transparent !important;
}

.shiki-wrapper :deep(code) {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: inherit;
}

.shiki-small {
  font-size: 11px;
  opacity: 0.85;
}

.shiki-tiny {
  font-size: 10.5px;
  opacity: 0.9;
}

.card-content-inner {
  display: flex;
  align-items: center;
  height: 100%;
}

.code-tiny {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 10.5px;
  color: inherit;
  opacity: 0.85;
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

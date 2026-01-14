<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { ShapeHistogramExample, SHAPE_HISTOGRAM_NUM_BINS } from "@/examples/ShapeHistogramExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";
import { useMeshSelection } from "@/composables/useMeshSelection";
import { getExampleMetadata } from "@/utils/liveExamples";
import { VisXYContainer, VisGroupedBar, VisAxis } from "@unovis/vue";

const metadata = getExampleMetadata("shape-histogram");
if (metadata) {
  defineOgImageComponent("Docs", {
    title: metadata.title,
    description: metadata.description,
    headline: "Live Example",
  });
  useSeoMeta({
    title: metadata.title,
    description: metadata.description,
  });
}

const { isTouchscreen } = useTouchscreen();
const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadExampleWithAssets } = useWasmModule();
const {
  isLoading,
  loadingMessage,
  loadingError,
  resetLoading,
  setLoadingMessage,
  failLoading,
  finishLoading,
} = useExampleLoadingState();
const { meshSize, buildMeshes, formatPolygonLabel } = useMeshSelection();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: ShapeHistogramExample | null = null;

const avgTime = ref("0");
const meshCount = 1;
const meshes = computed(() => buildMeshes(meshCount));
const polygonLabel = computed(() => formatPolygonLabel(meshCount));

const histogramBins = ref<number[]>(Array.from({ length: SHAPE_HISTOGRAM_NUM_BINS }, () => 0));
const hasHistogramData = computed(() => histogramBins.value.some((value) => value > 0));
const histogramData = computed(() =>
  histogramBins.value.map((value, index) => ({
    value,
    index,
  })),
);
const histogramX = (d: { index: number }) => d.index;
const histogramY = [(d: { value: number }) => d.value];
const histogramColor = () => "#00d5be";
const histogramMargin = { top: 8, right: 8, bottom: 4, left: 8 };
const histogramDomain = [-0.5, SHAPE_HISTOGRAM_NUM_BINS - 0.5] as [number, number];
const histogramTickValues = [
  histogramDomain[0],
  (histogramDomain[0] + histogramDomain[1]) / 2,
  histogramDomain[1],
];
const histogramTickFormat = (_value: number, i: number, ticks: number[]) =>
  i === 0 ? "-1" : i === ticks.length - 1 ? "1" : "0";

// Radius slider: 1% to 25% of AABB diagonal
const radiusPercent = ref(7.5);
const aabbDiagonal = ref(1);

const updateRadius = () => {
  if (exampleClass) {
    const radius = aabbDiagonal.value * (radiusPercent.value / 100);
    exampleClass.setRadius(radius);
  }
};

watch(radiusPercent, updateRadius);

let tearDownRequested = false;
let currentLoadId = 0;

const disposeExample = () => {
  if (exampleClass) {
    exampleClass.dispose();
    exampleClass = null;
  }
};

const loadThreejs = async () => {
  const loadId = ++currentLoadId;
  disposeExample();
  histogramBins.value = Array.from({ length: SHAPE_HISTOGRAM_NUM_BINS }, () => 0);
  exampleClass = await loadExampleWithAssets({
    meshes: meshes.value,
    skipOverlayIfCached: true,
    loading: { resetLoading, setLoadingMessage, failLoading, finishLoading },
    isTornDown: () => tearDownRequested || loadId !== currentLoadId,
    createScene: (wasmInstance, meshFilenames) => {
      const el = threejsContainer.value;
      if (!el) {
        return null;
      }

      const instance = new ShapeHistogramExample(wasmInstance, meshFilenames, el, isDark.value);
      instance.onHistogramUpdate = (bins) => {
        if (tearDownRequested || loadId !== currentLoadId) return;
        histogramBins.value = bins;
      };
      instance.refreshTimeValue = getAvgTime;
      // Store AABB diagonal for radius slider
      aabbDiagonal.value = instance.getAabbDiagonal();
      return instance;
    },
  });
};

const getAvgTime = () => {
  if (exampleClass) {
    avgTime.value = exampleClass.getAverageTime().toFixed(2);
  }
  return 0;
};

const badge = computed(() => ({
  icon: "i-lucide-gauge",
  label: "Last update:",
  value: `${avgTime.value} ms`,
  polygons: polygonLabel.value,
}));

watch(
  meshSize,
  () => {
    radiusPercent.value = 7.5;
    loadThreejs();
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  tearDownRequested = true;
  disposeExample();
});

watch(isDark, (dark) => {
  if (exampleClass) {
    exampleClass.applyTheme(dark);
  }
});
</script>
<template>
  <ExampleLayout
    :title="metadata?.title"
    :badge="badge"
    :polygon-label="polygonLabel"
    :loading="isLoading"
    :loading-message="loadingMessage"
    :loading-error="loadingError"
    @retry="loadThreejs"
  >
    <template #info>
      <ClientOnly>
        <div v-if="isTouchscreen" class="flex gap-1 items-center text-muted">
          <UIcon name="i-lucide-hand" class="size-4 ml-1" />
          <p class="text-sm">Touch and drag over the mesh to see local shape index histogram.</p>
        </div>
        <div v-else class="flex gap-1 items-center text-muted">
          <UIcon name="i-lucide-mouse-pointer" class="size-4 ml-1" />
          <p class="text-sm">Hover over the mesh to see local shape index histogram.</p>
        </div>
      </ClientOnly>
      <div class="flex gap-2 items-center text-muted mt-2">
        <UIcon name="i-lucide-circle" class="size-4 ml-1" />
        <span class="text-sm w-16">Radius:</span>
        <USlider v-model="radiusPercent" :min="0" :max="15" :step="0.5" class="w-32" />
        <span class="text-sm w-12">{{ radiusPercent.toFixed(1) }}%</span>
      </div>
    </template>
    <template #containers>
      <div class="relative h-full flex-1 min-h-0 w-screen md:w-full">
        <div ref="threejsContainer" id="threejsContainer" class="absolute inset-0"></div>
        <div
          class="pointer-events-none unovis absolute bottom-3 md:bottom-4 right-3 md:right-4 w-[calc(100vw-1.5rem)] sm:w-auto z-10 rounded-lg bg-neutral-400/10 shadow-lg backdrop-blur px-2.5 pt-2 pb-1.5 text-xs flex flex-col gap-1"
        >
          <div class="text-center text-sm md:text-base font-semibold tracking-wide opacity-90">
            Shape Index
          </div>
          <div class="h-[18vh] w-full sm:h-[20vh] sm:w-[20vw]">
            <Transition
              mode="out-in"
              enter-active-class="transition duration-200 ease-out"
              enter-from-class="opacity-0 translate-y-1"
              enter-to-class="opacity-100 translate-y-0"
              leave-active-class="transition duration-150 ease-in"
              leave-from-class="opacity-100 translate-y-0"
              leave-to-class="opacity-0 translate-y-1"
            >
              <div
                v-if="!hasHistogramData"
                key="empty"
                class="h-full w-full flex flex-col items-center justify-center gap-2 text-center text-sm sm:text-base text-muted px-3"
              >
                <ClientOnly>
                  <UIcon
                    :name="isTouchscreen ? 'i-lucide-hand' : 'i-lucide-mouse-pointer'"
                    class="size-6 md:size-8 text-primary opacity-80 animate-pulse"
                  />
                  <span v-if="isTouchscreen">
                    Touch and drag on the mesh to see the shape index histogram.
                  </span>
                  <span v-else>Hover over the mesh to see the shape index histogram.</span>
                </ClientOnly>
              </div>
              <div v-else key="chart" class="h-full w-full">
                <ClientOnly>
                  <VisXYContainer
                    class="h-full w-full"
                    :margin="histogramMargin"
                    :xDomain="histogramDomain"
                  >
                    <VisGroupedBar
                      :data="histogramData"
                      :x="histogramX"
                      :y="histogramY"
                      :color="histogramColor"
                      :duration="50"
                    />
                    <VisAxis
                      type="x"
                      :tickValues="histogramTickValues"
                      :tickFormat="histogramTickFormat"
                      :gridLine="false"
                      :tickLine="false"
                      :tickPadding="1"
                    />
                  </VisXYContainer>
                </ClientOnly>
              </div>
            </Transition>
          </div>
        </div>
      </div>
    </template>
  </ExampleLayout>
</template>

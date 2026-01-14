<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { ShapeHistogramExample } from "@/examples/ShapeHistogramExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";
import { useMeshSelection } from "@/composables/useMeshSelection";
import { getExampleMetadata } from "@/utils/liveExamples";

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
const { isLoading, loadingMessage, loadingError, resetLoading, setLoadingMessage, failLoading, finishLoading } =
  useExampleLoadingState();
const { meshSize, buildMeshes, formatPolygonLabel } = useMeshSelection();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: ShapeHistogramExample | null = null;

const avgTime = ref("0");
const meshCount = 1;
const meshes = computed(() => buildMeshes(meshCount));
const polygonLabel = computed(() => formatPolygonLabel(meshCount));

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

      const instance = new ShapeHistogramExample(
        wasmInstance,
        meshFilenames,
        el,
        isDark.value,
      );
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

watch(meshSize, () => loadThreejs(), { immediate: true });

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
      <div v-if="isTouchscreen" class="flex gap-1 items-center text-muted">
        <UIcon name="i-lucide-hand" class="size-4 ml-1" />
        <p class="text-sm">Touch and drag over the mesh to see local shape index histogram.</p>
      </div>
      <div v-else class="flex gap-1 items-center text-muted">
        <UIcon name="i-lucide-mouse-pointer" class="size-4 ml-1" />
        <p class="text-sm">Hover over the mesh to see local shape index histogram.</p>
      </div>
      <div class="flex gap-2 items-center text-muted mt-2">
        <UIcon name="i-lucide-circle" class="size-4 ml-1" />
        <span class="text-sm w-16">Radius:</span>
        <USlider v-model="radiusPercent" :min="0" :max="15" :step="0.5" class="w-32" />
        <span class="text-sm w-12">{{ radiusPercent.toFixed(1) }}%</span>
      </div>
    </template>
    <template #containers>
      <div ref="threejsContainer" id="threejsContainer" class="h-full flex-1 min-h-0 w-screen md:w-full"></div>
    </template>
  </ExampleLayout>
</template>

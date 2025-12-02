<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { FormsIntersectionsExample } from "@/examples/FormsIntersectionsExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadExampleWithAssets } = useWasmModule();
const { isLoading, loadingMessage, loadingError, resetLoading, setLoadingMessage, failLoading, finishLoading } =
  useExampleLoadingState();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: FormsIntersectionsExample | null = null;

const avgTime = ref("0");
const totalPolygons = ref("0");
const meshes = [
  { url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" },
  { url: "/stl/Stanford_Bunny.stl", filename: "Stanford_Bunny.stl" },
];

let tearDownRequested = false;
const loadThreejs = async () => {
  exampleClass = await loadExampleWithAssets({
    meshes,
    skipOverlayIfCached: true,
    loading: { resetLoading, setLoadingMessage, failLoading, finishLoading },
    isTornDown: () => tearDownRequested,
    createScene: (wasmInstance, meshFilenames) => {
      const el = threejsContainer.value;
      if (!el) {
        return null;
      }

      const instance = new FormsIntersectionsExample(
        wasmInstance,
        meshFilenames,
        el,
        isDark.value,
      );
      totalPolygons.value = wasmInstance.get_number_of_polygons().toString();
      instance.refreshTimeValue = getAvgTime;
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
  label: "Curve update:",
  value: `${avgTime.value} ms`,
}));

const actionButtons = [
  { icon: "i-lucide-rotate-3d", label: "Randomize", keyboardShortcut: "N", onClick: () => exampleClass?.randomize() },
];

onMounted(() => {
  loadThreejs();
});

onBeforeUnmount(() => {
  tearDownRequested = true;
  if (exampleClass) {
    exampleClass.dispose();
    exampleClass = null;
  }
});

watch(isDark, (dark) => {
  if (exampleClass) {
    exampleClass.applyTheme(dark);
  }
});
</script>

<template>
  <div class="flex flex-col w-full h-full">
    <div class="flex flex-row flex-1 relative min-h-0">
      <ExampleInfoCard title="Mesh Intersections" :badge="badge">
        <div class="flex gap-2 items-center text-muted">
          <UIcon name="i-lucide-hand" class="size-4 ml-1" />
          <p class="text-sm">Drag a mesh. The intersection curves recompute instantly.</p>
        </div>
        <div class="grid grid-cols-1 gap-1 text-muted">
          <p class="text-sm">Total polygons: {{ totalPolygons }}</p>
        </div>
      </ExampleInfoCard>
      <div class="flex flex-col md:flex-row w-full">
        <div ref="threejsContainer" id="threejsContainer" class="h-full flex-1 min-h-0 w-[100vw] md:w-full"></div>
      </div>
      <ExampleLoadingOverlay :loading="isLoading" :message="loadingMessage" :error="loadingError" @retry="loadThreejs" />
      <ExampleActionButtons :buttons="actionButtons" />
    </div>
  </div>
</template>

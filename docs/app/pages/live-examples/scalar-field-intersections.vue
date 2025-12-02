<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { ScalarFieldIntersectionsExample } from "@/examples/ScalarFieldIntersectionsExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadExampleWithAssets } = useWasmModule();
const { isLoading, loadingMessage, loadingError, resetLoading, setLoadingMessage, failLoading, finishLoading } =
  useExampleLoadingState();

const { isTouchscreen } = useTouchscreen();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: ScalarFieldIntersectionsExample | null = null;

const avgTime = ref("0");
const meshes = [{ url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" }];

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

      const instance = new ScalarFieldIntersectionsExample(
        wasmInstance,
        meshFilenames,
        el,
        isDark.value,
      );
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
  label: "Last scroll:",
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
      <ExampleInfoCard title="Scalar Field Intersections" :badge="badge">
        <div v-if="isTouchscreen" class="flex gap-1 items-center text-muted">
          <UIcon name="i-lucide-tally-3" class="size-4 ml-1" />
          <p class="text-sm">Drag the top plane with 3 fingers to sweep.</p>
        </div>
        <div v-else class="flex gap-1 items-center text-muted">
          <UKbd variant="soft" value="shift" />
          <UKbd variant="soft" value="scroll" />
          <p class="text-sm">Scroll to move the plane. Contour lines update live.</p>
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

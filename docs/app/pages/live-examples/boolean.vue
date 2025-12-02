<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { BooleanExample } from "@/examples/BooleanExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadExampleWithAssets } = useWasmModule();
const { isLoading, loadingMessage, loadingError, resetLoading, setLoadingMessage, failLoading, finishLoading } =
  useExampleLoadingState();

const threejsContainer = ref<HTMLElement | null>(null);
const threejsContainer2 = ref<HTMLElement | null>(null);
let exampleClass: BooleanExample | null = null;
const meshes = [
  { url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" },
  { url: "/stl/Stanford_Bunny.stl", filename: "Stanford_Bunny.stl" },
];

const avgTime = ref("0");
const getAvgTime = () => {
  if (exampleClass) {
    avgTime.value = exampleClass.getAverageTime().toFixed(2);
  }
  return 0;
};

const badge = computed(() => ({
  icon: "i-lucide-gauge",
  label: "Last boolean:",
  value: `${avgTime.value} ms`,
}));

const actionButtons = [
  { icon: "i-lucide-rotate-3d", label: "Randomize", keyboardShortcut: "N", onClick: () => exampleClass?.randomize() },
  { icon: "i-lucide-focus", label: "Resync camera", keyboardShortcut: "R", onClick: () => exampleClass?.resyncCamera() },
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
      const el2 = threejsContainer2.value;
      if (!el || !el2) {
        return null;
      }

      const instance = new BooleanExample(
        wasmInstance,
        meshFilenames,
        el,
        el2,
        isDark.value,
      );
      instance.refreshTimeValue = getAvgTime;
      return instance;
    },
  });
};

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
      <ExampleInfoCard v-if="!isLoading" title="Boolean" :badge="badge">
        <div class="flex gap-2 items-center text-muted">
          <UIcon name="i-lucide-hand" class="size-4 ml-1" />
          <p class="text-sm">Drag a mesh. The boolean updates in real time.</p>
        </div>
      </ExampleInfoCard>
      <div class="flex flex-col md:flex-row w-full">
        <div ref="threejsContainer" id="threejsContainer" class="h-full flex-1 min-h-0 w-[100vw] md:w-full"></div>
        <div ref="threejsContainer2" id="threejsContainer2" class="h-full flex-1 min-h-0 w-[100vw] md:w-full"></div>
      </div>
      <ExampleLoadingOverlay :loading="isLoading" :message="loadingMessage" :error="loadingError" @retry="loadThreejs" />
      <ExampleActionButtons :buttons="actionButtons" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { PositioningExample } from "@/examples/PositioningExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadExampleWithAssets } = useWasmModule();
const { isLoading, loadingMessage, loadingError, resetLoading, setLoadingMessage, failLoading, finishLoading } =
  useExampleLoadingState();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: PositioningExample | null = null;
const meshes = [
  { url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" },
  { url: "/stl/Stanford_Bunny.stl", filename: "Stanford_Bunny.stl" },
];

const actionButtons = [
  { icon: "i-lucide-rotate-3d", label: "Randomize", keyboardShortcut: "N", onClick: () => exampleClass?.randomize() },
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

      return new PositioningExample(
        wasmInstance,
        meshFilenames,
        el,
        isDark.value,
      );
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
      <ExampleInfoCard title="Positioning">
        <div class="flex gap-2 items-center text-muted">
          <UIcon name="i-lucide-hand" class="size-4 ml-1" />
          <p class="text-sm">Drag a mesh and release. It snaps to the nearest point instantly.</p>
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

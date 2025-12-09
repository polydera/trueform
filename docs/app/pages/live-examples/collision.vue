<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { CollisionExample } from "@/examples/CollisionExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";
import { useMeshSelection } from "@/composables/useMeshSelection";
import { getExampleMetadata } from "@/utils/liveExamples";

const metadata = getExampleMetadata("collision");
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
let exampleClass: CollisionExample | null = null;

const avgTime = ref("0");
const avgPickTime = ref("0");
const meshCount = 25; // 5x5 grid as defined in collision_web.h
const meshes = computed(() => buildMeshes(2)); // Only 2 unique meshes are loaded
const polygonLabel = computed(() => formatPolygonLabel(meshCount));

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

      const instance = new CollisionExample(wasmInstance, meshFilenames, el, isDark.value);
      instance.refreshTimeValue = getAvgTime;
      return instance;
    },
  });
};

const getAvgTime = () => {
  if (exampleClass) {
    avgTime.value = exampleClass.getAverageTime().toFixed(2);
    avgPickTime.value = exampleClass.getAveragePickTime().toFixed(2);
  }
  return 0;
};

const badge = computed(() => ({
  icon: "i-lucide-gauge",
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
      <div class="flex gap-2 items-center text-muted">
        <UIcon name="i-lucide-hand" class="size-4 ml-1" />
        <p class="text-sm">Drag a mesh. Contact detection runs live as you move.</p>
      </div>
    </template>
    <template #containers>
      <div
        ref="threejsContainer"
        id="threejsContainer"
        class="h-full flex-1 min-h-0 w-screen md:w-full"
      ></div>
    </template>
  </ExampleLayout>
</template>

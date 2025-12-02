<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { CollisionExample } from "@/examples/CollisionExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";
import { useMeshSelection } from "@/composables/useMeshSelection";

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
const meshCount = 2;
const meshes = computed(() => buildMeshes(meshCount));
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
    avgTime.value = (exampleClass.getAverageTime() * 1000).toFixed(2);
    avgPickTime.value = (exampleClass.getAveragePickTime() * 1000).toFixed(2);
  }
  return 0;
};

const badge = computed(() => ({
  icon: "i-lucide-gauge",
  label: "",
  value: `Pick: ${avgPickTime.value} μs, Collision: ${avgTime.value} μs`,
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
  <div class="flex flex-col w-full h-full">
    <div class="flex flex-row flex-1 relative min-h-0">
      <ExampleInfoCard title="Collision" :badge="badge">
        <div class="flex gap-2 items-center text-muted">
          <UIcon name="i-lucide-hand" class="size-4 ml-1" />
          <p class="text-sm">Drag a mesh. Contact detection runs live as you move.</p>
        </div>
      </ExampleInfoCard>
      <div class="flex flex-col md:flex-row w-full">
        <div
          ref="threejsContainer"
          id="threejsContainer"
          class="h-full flex-1 min-h-0 w-[100vw] md:w-full"
        ></div>
      </div>
      <ExampleLoadingOverlay
        :loading="isLoading"
        :message="loadingMessage"
        :error="loadingError"
        @retry="loadThreejs"
      />
      <ExamplePolygonsCard
        :mesh-count="meshCount"
        :mesh-label="polygonLabel"
        :loading="isLoading"
      />
    </div>
  </div>
</template>

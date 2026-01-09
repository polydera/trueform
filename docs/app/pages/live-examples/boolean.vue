<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { BooleanExample } from "@/examples/BooleanExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";
import { useMeshSelection } from "@/composables/useMeshSelection";
import { getExampleMetadata } from "@/utils/liveExamples";

const metadata = getExampleMetadata("boolean");
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
const threejsContainer2 = ref<HTMLElement | null>(null);
let exampleClass: BooleanExample | null = null;
const meshCount = 1;
const meshes = computed(() => buildMeshes(meshCount));
const polygonLabel = computed(() => formatPolygonLabel(meshCount));
const sphereSizeSteps = ref(0);
const sphereSizeBounds = {
  min: -24,
  max: 50,
  step: 1,
};
const isSyncingSphereSize = ref(false);

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
  polygons: polygonLabel.value,
}));

const actionButtons = [
  {
    icon: "i-lucide-focus",
    label: "Resync camera",
    keyboardShortcut: "R",
    onClick: () => exampleClass?.resyncCamera(),
  },
];

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
  sphereSizeSteps.value = 0;
  exampleClass = await loadExampleWithAssets({
    meshes: meshes.value,
    skipOverlayIfCached: true,
    loading: { resetLoading, setLoadingMessage, failLoading, finishLoading },
    isTornDown: () => tearDownRequested || loadId !== currentLoadId,
    createScene: (wasmInstance, meshFilenames) => {
      const el = threejsContainer.value;
      const el2 = threejsContainer2.value;
      if (!el || !el2) {
        return null;
      }

      const instance = new BooleanExample(wasmInstance, meshFilenames, el, el2, isDark.value);
      instance.refreshTimeValue = getAvgTime;
      instance.onSphereSizeDelta = (delta) => {
        const nextValue = Math.min(
          sphereSizeBounds.max,
          Math.max(sphereSizeBounds.min, sphereSizeSteps.value + delta),
        );
        if (nextValue === sphereSizeSteps.value) return;
        isSyncingSphereSize.value = true;
        sphereSizeSteps.value = nextValue;
      };
      return instance;
    },
  });
};

watch(meshSize, () => loadThreejs(), { immediate: true });
watch(sphereSizeSteps, (value, oldValue) => {
  if (!exampleClass) return;
  if (isSyncingSphereSize.value) {
    isSyncingSphereSize.value = false;
    return;
  }
  const delta = value - oldValue;
  if (delta !== 0) {
    exampleClass.adjustSphereSize(delta);
  }
});

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
    :action-buttons="actionButtons"
    hide-info-during-loading
    @retry="loadThreejs"
  >
    <template #info>
      <div class="flex gap-2 items-center text-muted">
        <UIcon name="i-lucide-hand" class="size-4 ml-1" />
        <p class="text-sm">Drag a mesh. The boolean updates in real time.</p>
      </div>
      <div class="flex flex-col gap-2">
        <div class="flex items-center justify-between gap-3 text-sm text-muted w-full">
          <div class="min-w-30 flex items-center gap-2">
            <UIcon name="i-lucide-scaling" class="size-4 ml-1" />
            Sphere size
          </div>
          <USlider
            v-model="sphereSizeSteps"
            :min="sphereSizeBounds.min"
            :max="sphereSizeBounds.max"
            :step="sphereSizeBounds.step"
          />
        </div>
      </div>
    </template>
    <template #containers>
      <div
        ref="threejsContainer"
        id="threejsContainer"
        class="h-full flex-1 min-h-0 w-screen md:w-full"
      ></div>
      <div
        ref="threejsContainer2"
        id="threejsContainer2"
        class="h-full flex-1 min-h-0 w-screen md:w-full"
      ></div>
    </template>
  </ExampleLayout>
</template>

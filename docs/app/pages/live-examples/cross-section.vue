<script setup lang="ts">
import { useTrueform } from "@/composables/useTrueform";
import { CrossSectionExample } from "@/examples/CrossSectionExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";
import { useMeshSelection } from "@/composables/useMeshSelection";
import { getExampleMetadata } from "@/utils/liveExamples";

const metadata = getExampleMetadata("cross-section");
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
const { load: loadTF } = useTrueform();
const { isLoading, loadingMessage, loadingError, resetLoading, setLoadingMessage, failLoading, finishLoading } =
  useExampleLoadingState();
const { meshSize, meshUrl, meshFilename, formatPolygonLabel } = useMeshSelection();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: CrossSectionExample | null = null;
const polygonLabel = computed(() => formatPolygonLabel(1));

const avgTime = ref("0");
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
  polygons: polygonLabel.value,
}));

const actionButtons = [
  { icon: "i-lucide-rotate-3d", label: "Randomize", keyboardShortcut: "N", onClick: () => exampleClass?.randomize() },
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
  resetLoading();

  try {
    setLoadingMessage("Loading trueform...");
    const tf = await loadTF();
    if (tearDownRequested || loadId !== currentLoadId) { finishLoading(); return; }

    setLoadingMessage("Fetching mesh...");
    const resp = await fetch(meshUrl.value);
    const fileBuffer = await resp.arrayBuffer();
    if (tearDownRequested || loadId !== currentLoadId) { finishLoading(); return; }

    setLoadingMessage("Initializing renderer...");
    const el = threejsContainer.value;
    if (!el) { finishLoading(); return; }

    exampleClass = new CrossSectionExample(tf, fileBuffer, meshFilename.value, el, isDark.value);
    exampleClass.refreshTimeValue = getAvgTime;
    getAvgTime();
    finishLoading();
  } catch (error) {
    if (!tearDownRequested && loadId === currentLoadId) {
      failLoading(error);
    }
  }
};

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
    :action-buttons="actionButtons"
    @retry="loadThreejs"
  >
    <template #info>
      <div v-if="isTouchscreen" class="flex gap-1 items-center text-muted">
        <UIcon name="i-lucide-hand" class="size-4 ml-1" />
        <p class="text-sm">Scroll with one finger to move the cutting plane.</p>
      </div>
      <div v-else class="flex gap-1 items-center text-muted">
        <UKbd variant="soft" value="scroll" />
        <p class="text-sm">Scroll to move the cutting plane. Cross-sections update live.</p>
      </div>
    </template>
    <template #containers>
      <div ref="threejsContainer" id="threejsContainer" class="h-full w-full min-h-0"></div>
    </template>
  </ExampleLayout>
</template>

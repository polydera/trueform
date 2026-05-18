<script setup lang="ts">
import { useTrueform } from "@/composables/useTrueform";
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
const { load: loadTF } = useTrueform();
const { isLoading, loadingMessage, loadingError, resetLoading, setLoadingMessage, failLoading, finishLoading } =
  useExampleLoadingState();
const { meshSize, meshUrl, meshFilename, formatPolygonLabel } = useMeshSelection();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: CollisionExample | null = null;

const meshCount = 25; // 5x5 grid
const polygonLabel = computed(() => formatPolygonLabel(meshCount));

const avgTime = ref("0");
const avgPickTime = ref("0");
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

    exampleClass = new CollisionExample(tf, fileBuffer, meshFilename.value, el, isDark.value);
    exampleClass.refreshTimeValue = getAvgTime;
    getAvgTime();
    finishLoading();
  } catch (error) {
    if (!tearDownRequested && loadId === currentLoadId) {
      failLoading(error);
    }
  }
};

onMounted(() => loadThreejs());
watch(meshSize, () => loadThreejs());

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

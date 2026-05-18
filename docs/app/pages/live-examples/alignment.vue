<script setup lang="ts">
import { useTrueform } from "@/composables/useTrueform";
import { AlignmentExample, type InteractionMode } from "@/examples/AlignmentExample";
import { useExampleLoadingState } from "@/composables/useExampleLoadingState";
import { useMeshSelection } from "@/composables/useMeshSelection";
import { getExampleMetadata } from "@/utils/liveExamples";

const metadata = getExampleMetadata("alignment");
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
const {
  isLoading,
  loadingMessage,
  loadingError,
  resetLoading,
  setLoadingMessage,
  failLoading,
  finishLoading,
} = useExampleLoadingState();
const { meshSize, meshUrl, meshFilename } = useMeshSelection();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: AlignmentExample | null = null;

const polygonLabel = computed(() => `${meshSize.value} + 50k`);

const alignmentTime = ref(0);
const isAligned = ref(false);
const currentMode = ref<InteractionMode>("move");

const badge = computed(() => {
  if (isAligned.value) {
    return {
      icon: "i-lucide-check-circle",
      label: "Mesh Registration:",
      value: `${alignmentTime.value.toFixed(1)} ms`,
      polygons: polygonLabel.value,
    };
  }
  return {
    icon: "i-lucide-locate",
    label: "Mesh Registration",
    polygons: polygonLabel.value,
  };
});

const handleAlign = () => {
  if (!exampleClass) return;
  const time = exampleClass.align();
  alignmentTime.value = time;
  isAligned.value = true;
};

const setMode = (mode: InteractionMode) => {
  if (!exampleClass) return;
  currentMode.value = mode;
  exampleClass.setMode(mode);
};

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
  isAligned.value = false;
  alignmentTime.value = 0;
  currentMode.value = "move";
  resetLoading();

  try {
    setLoadingMessage("Loading trueform...");
    const tf = await loadTF();
    if (tearDownRequested || loadId !== currentLoadId) { finishLoading(); return; }

    setLoadingMessage("Fetching meshes...");
    const [sourceResp, targetResp] = await Promise.all([
      fetch(meshUrl.value),
      fetch("/stl/dragon-50k.stl"),
    ]);
    const [sourceBuffer, targetBuffer] = await Promise.all([
      sourceResp.arrayBuffer(),
      targetResp.arrayBuffer(),
    ]);
    if (tearDownRequested || loadId !== currentLoadId) { finishLoading(); return; }

    setLoadingMessage("Initializing renderer...");
    const el = threejsContainer.value;
    if (!el) { finishLoading(); return; }

    exampleClass = new AlignmentExample(
      tf, sourceBuffer, meshFilename.value, targetBuffer, el, isDark.value,
    );
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
    hide-info-during-loading
    @retry="loadThreejs"
  >
    <template #info>
      <div class="flex gap-2 items-center text-muted">
        <UIcon name="i-lucide-hand" class="size-4 ml-1" />
        <p class="text-sm">Move and Rotate the mesh, then click Align.</p>
      </div>
      <div class="flex items-center justify-between mt-2">
        <UButtonGroup size="sm">
          <UButton
            icon="i-lucide-move"
            :variant="currentMode === 'move' ? 'solid' : 'ghost'"
            @click="setMode('move')"
          >
            Move
          </UButton>
          <UButton
            icon="i-lucide-rotate-3d"
            :variant="currentMode === 'rotate' ? 'solid' : 'ghost'"
            @click="setMode('rotate')"
          >
            Rotate
          </UButton>
        </UButtonGroup>
        <UButtonGroup size="sm">
          <UButton icon="i-lucide-target" color="primary" @click="handleAlign">
            Align
          </UButton>
        </UButtonGroup>
      </div>
    </template>
    <template #containers>
      <div class="relative h-full flex-1 min-h-0 w-screen md:w-full">
        <div ref="threejsContainer" id="threejsContainer" class="absolute inset-0"></div>
      </div>
    </template>
  </ExampleLayout>
</template>

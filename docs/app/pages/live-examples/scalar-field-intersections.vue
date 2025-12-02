<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { ScalarFieldIntersectionsExample } from "@/examples/ScalarFieldIntersectionsExample";

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadWasmModule, preloadMeshes } = useWasmModule();

const { isTouchscreen } = useTouchscreen();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: ScalarFieldIntersectionsExample | null = null;

const avgTime = ref("0");
const meshes = [{ url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" }];

let tearDownRequested = false;

const loadThreejs = async () => {
  const wasmInstance = await loadWasmModule();
  if (tearDownRequested) return;

  const el = threejsContainer.value;
  if (!el) return;

  await preloadMeshes(wasmInstance, meshes);

  if (tearDownRequested) return;

  exampleClass = new ScalarFieldIntersectionsExample(
    wasmInstance,
    meshes.map((m) => m.filename),
    el,
    isDark.value,
  );
  exampleClass.refreshTimeValue = getAvgTime;
};
const getAvgTime = () => {
  if (exampleClass) {
    avgTime.value = exampleClass.getAverageTime().toFixed(2);
  }
  return 0;
};

const badge = computed(() => ({
  icon: "i-lucide-gauge",
  text: `Last scroll: ${avgTime.value} ms`,
}));

const actionButtons = [
  { icon: "i-lucide-rotate-3d", label: "Randomize", keyboardShortcut: "N" },
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
          <p class="text-sm">Drag the plane with 3 fingers to sweep.</p>
        </div>
        <div v-else class="flex gap-3 items-center text-muted">
          <UIcon name="i-lucide-info" class="size-4 ml-1" />
          <p class="text-sm">Sweep the plane and isobands.</p>
        </div>
      </ExampleInfoCard>
      <div class="flex flex-col md:flex-row w-full">
        <div ref="threejsContainer" id="threejsContainer" class="h-full flex-1 min-h-0 w-[100vw] md:w-full"></div>
      </div>
      <ExampleActionButtons :buttons="actionButtons" />
    </div>
  </div>
</template>

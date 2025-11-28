<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { IsobandsExample } from "@/examples/IsobandsExample";

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadWasmModule, preloadMeshes } = useWasmModule();

const threejsContainer = ref<HTMLElement | null>(null);
const threejsContainer2 = ref<HTMLElement | null>(null);
let exampleClass: IsobandsExample | null = null;
const meshes = [{ url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" }];

const avgTime = ref("0");
const getAvgTime = () => {
  if (exampleClass) {
    avgTime.value = exampleClass.getAverageTime().toFixed(2);
  }
  return 0;
};

let tearDownRequested = false;

const loadThreejs = async () => {
  const wasmInstance = await loadWasmModule();
  if (tearDownRequested) return;

  const el = threejsContainer.value;
  const el2 = threejsContainer2.value;
  if (!el || !el2) return;

  await preloadMeshes(wasmInstance, meshes);

  if (tearDownRequested) return;

  exampleClass = new IsobandsExample(
    wasmInstance,
    meshes.map((m) => m.filename),
    el,
    el2,
    isDark.value,
  );
  exampleClass.refreshTimeValue = getAvgTime;
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
  <div class="flex flex-col w-full">
    <div class="flex flex-row flex-1 relative">
      <div
        class="absolute left-3 top-3 z-10 max-w-md rounded-lg p-3 bg-neutral-100/10 shadow-lg backdrop-blur"
      >
        <p class="font-semibold text-lg mb-2">Isobands</p>
        <div class="flex flex-col gap-2 text-sm">
          <div class="flex gap-1 items-center text-muted">
            <UKbd variant="subtle" value="shift"/><UKbd variant="subtle">Scroll</UKbd>
            <p>Sweep the plane and isobands.</p>
          </div>
          <div class="flex gap-2 items-center text-muted">
            <UKbd variant="subtle">r</UKbd>
            <p class="text-sm">Resync camera controls</p>
          </div>
          <div class="flex gap-1.5 items-center text-muted">
            <UKbd variant="subtle">n</UKbd>
            <p>Randomize plane orientation</p>
          </div>
          <div class="flex gap-2 items-center text-muted">
            <UIcon name="i-lucide-gauge" class="size-4 ml-1" />
            <p class="text-sm">Last scroll: {{ avgTime }} ms</p>
          </div>
        </div>
      </div>
      <div ref="threejsContainer" id="threejsContainer" class="h-full w-full m-0 p-0"></div>
      <div ref="threejsContainer2" id="threejsContainer2" class="h-full w-full m-0 p-0"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { BooleanExample } from "@/examples/BooleanExample";

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadWasmModule, preloadMeshes } = useWasmModule();

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
  text: `Last boolean: ${avgTime.value} ms`,
}));

const actionButtons = [
  { icon: "i-lucide-rotate-3d", label: "Randomize", keyboardShortcut: "N", onClick: () => exampleClass?.randomize() },
  { icon: "i-lucide-focus", label: "Resync camera", keyboardShortcut: "R", onClick: () => exampleClass?.resyncCamera() },
];

let tearDownRequested = false;

const loadThreejs = async () => {
  const wasmInstance = await loadWasmModule();
  if (tearDownRequested) return;

  const el = threejsContainer.value;
  const el2 = threejsContainer2.value;
  if (!el || !el2) return;

  await preloadMeshes(wasmInstance, meshes);

  if (tearDownRequested) return;

  exampleClass = new BooleanExample(
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
  <div class="flex flex-col w-full h-full">
    <div class="flex flex-row flex-1 relative min-h-0">
      <ExampleInfoCard title="Boolean" :badge="badge">
        <div class="flex gap-3 items-center text-muted">
          <UIcon name="i-lucide-info" class="size-4 ml-1" />
          <p class="text-sm">Grab a mesh and move it to inspect intersection curve and difference mesh.</p>
        </div>
      </ExampleInfoCard>
      <div class="flex flex-col md:flex-row w-full">
        <div ref="threejsContainer" id="threejsContainer" class="h-full flex-1 min-h-0 w-[100vw] md:w-full"></div>
        <div ref="threejsContainer2" id="threejsContainer2" class="h-full flex-1 min-h-0 w-[100vw] md:w-full"></div>
      </div>
      <ExampleActionButtons :buttons="actionButtons" />
    </div>
  </div>
</template>

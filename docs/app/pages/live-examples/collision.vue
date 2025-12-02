<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { CollisionExample } from "@/examples/CollisionExample";

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadWasmModule, preloadMeshes } = useWasmModule();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: CollisionExample | null = null;

const totalPolygons = ref("0");
const avgTime = ref("0");
const avgPickTime = ref("0");
const meshes = [
  { url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" },
  { url: "/stl/Stanford_Bunny.stl", filename: "Stanford_Bunny.stl" },
];

let tearDownRequested = false;

const loadThreejs = async () => {
  const wasmInstance = await loadWasmModule();
  if (tearDownRequested) return;

  const el = threejsContainer.value;
  if (!el) return;

  await preloadMeshes(wasmInstance, meshes);

  if (tearDownRequested) return;

  exampleClass = new CollisionExample(
    wasmInstance,
    meshes.map((m) => m.filename),
    el,
    isDark.value,
  );
  totalPolygons.value = wasmInstance.get_number_of_polygons().toString();
  exampleClass.refreshTimeValue = getAvgTime;
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
  text: `Pick: ${avgPickTime.value} μs, Collision: ${avgTime.value} μs`,
}));

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
      <ExampleInfoCard title="Collision" :badge="badge">
        <div class="flex gap-3 items-center text-muted">
          <UIcon name="i-lucide-move-3d" class="size-4 ml-1" />
          <p class="text-sm">Grab and drag a mesh to test. Intersections highlight in real time.</p>
        </div>
        <div class="grid grid-cols-1 gap-1 text-muted">
          <p class="text-sm">Total polygons: {{ totalPolygons }}</p>
        </div>
      </ExampleInfoCard>
      <div class="flex flex-col md:flex-row w-full">
        <div ref="threejsContainer" id="threejsContainer" class="h-full flex-1 min-h-0 w-[100vw] md:w-full"></div>
      </div>
    </div>
  </div>
</template>


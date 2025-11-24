<script setup lang="ts">
import WASM from "@/examples/native";
import type { MainModule } from "@/examples/native";
import { FormsIntersectionsExample } from "@/examples/FormsIntersectionsExample";

let wasmInstance: MainModule | null = null;
const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");

const threejsContainer = ref();
let exampleClass: FormsIntersectionsExample;

const avgTime = ref("0");
const totalPolygons = ref("0");

const loadWasm = async () => {
  wasmInstance = await WASM();
};

const loadThreejs = async () => {
  if (wasmInstance === null) {
    await loadWasm();
  }
  const el = document.getElementById("threejsContainer");
  if (el && wasmInstance) {
    const meshes = [
      { url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" },
      { url: "/stl/Stanford_Bunny.stl", filename: "Stanford_Bunny.stl" },
    ];

    for (const mesh of meshes) {
      const response = await fetch(mesh.url);
      const buffer = await response.arrayBuffer();
      wasmInstance.FS.writeFile(mesh.filename, new Int8Array(buffer));
    }

    exampleClass = new FormsIntersectionsExample(
      wasmInstance,
      meshes.map((m) => m.filename),
      el,
      isDark.value,
    );
    totalPolygons.value = wasmInstance.get_number_of_polygons().toString();
  }
};

const getAvgTime = () => {
  if (exampleClass) {
    avgTime.value = exampleClass.getAverageTime().toFixed(2);
  }
  return 0;
};
setInterval(getAvgTime, 1000);

onMounted(() => {
  loadThreejs();
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
      <div class="absolute left-3 top-3 z-10 max-w-md rounded-lg p-3 bg-neutral-100/10 shadow-lg backdrop-blur">
        <p class="font-semibold text-lg mb-2">Forms Intersections</p>
        <div class="flex flex-col gap-2 text-sm text-muted">
          <div class="flex gap-3 items-center">
            <UIcon name="i-lucide-git-merge" class="size-4 ml-1" />
            <p>Grab and drag meshes; intersection curves update live.</p>
          </div>
          <div class="flex gap-2 items-center">
            <UKbd variant="subtle">n</UKbd>
            <p>Randomize mesh orientation</p>
          </div>
          <div class="grid grid-cols-1 gap-1">
            <p>Total polygons: {{ totalPolygons }}</p>
            <p>Curve update: {{ avgTime }} ms</p>
          </div>
        </div>
      </div>
      <div ref="threejsContainer" id="threejsContainer" class="h-full w-full m-0 p-0"></div>
    </div>
  </div>
</template>

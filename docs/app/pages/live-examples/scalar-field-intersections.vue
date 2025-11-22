<script setup lang="ts">
import WASM from "@/examples/native";
import type { MainModule } from "@/examples/native";
import { ScalarFieldIntersectionsExample } from "@/examples/ScalarFieldIntersectionsExample";

let wasmInstance: MainModule | null = null;
const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");

const threejsContainer = ref();
let exampleClass: ScalarFieldIntersectionsExample;

const avgTime = ref("0");

const loadWasm = async () => {
  wasmInstance = await WASM();
};

const loadThreejs = async () => {
  if (wasmInstance === null) {
    await loadWasm();
  }
  const el = document.getElementById("threejsContainer");
  if (el && wasmInstance) {
    const meshes = [{ url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" }];
    for (const mesh of meshes) {
      const response = await fetch(mesh.url);
      const buffer = await response.arrayBuffer();
      wasmInstance.FS.writeFile(mesh.filename, new Int8Array(buffer));
    }
    exampleClass = new ScalarFieldIntersectionsExample(
      wasmInstance,
      meshes.map((m) => m.filename),
      el,
      isDark.value,
    );
  }
};

const getAvgTime = () => {
  if (exampleClass) {
    avgTime.value = exampleClass.getAverageTime().toFixed(2);
  }
  return 0;
};

let avgTimer: ReturnType<typeof setInterval> | null = null;

onMounted(() => {
  loadThreejs();
  avgTimer = setInterval(getAvgTime, 1000);
});

onBeforeUnmount(() => {
  if (avgTimer) {
    clearInterval(avgTimer);
  }
});

watch(isDark, (dark) => {
  if (exampleClass) {
    exampleClass.applyTheme(dark);
  }
});
</script>

<template>
  <div class="flex flex-col w-full mt-(--ui-header-height)">
    <div class="flex flex-row flex-1 live-example-stage relative">
      <div class="absolute left-3 top-3 z-10 max-w-md rounded-lg p-3 bg-neutral-100/10 shadow-lg backdrop-blur">
        <p class="font-semibold text-lg mb-2">Scalar Field Intersections</p>
        <div class="flex flex-col gap-2 text-sm text-muted">
          <div class="flex gap-3 items-center">
            <UIcon name="i-lucide-mouse-pointer-2" class="size-4 ml-1" />
            <p>Hold shift and scroll to move the scalar field plane.</p>
          </div>
          <div class="flex gap-2 items-center">
            <UKbd variant="subtle">n</UKbd>
            <p>Randomize plane orientation</p>
          </div>
          <div class="flex gap-2 items-center">
            <UIcon name="i-lucide-gauge" class="size-4 ml-1" />
            <p>Last scroll: {{ avgTime }} ms</p>
          </div>
        </div>
      </div>
      <div ref="threejsContainer" id="threejsContainer" class="h-full w-full m-0 p-0"></div>
    </div>
  </div>
</template>

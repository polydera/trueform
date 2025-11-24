<script setup lang="ts">
import WASM from "@/examples/native";
import type { MainModule } from "@/examples/native";
import { IsobandsExample } from "@/examples/IsobandsExample";

let wasmInstance: MainModule | null = null;
const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");

const threejsContainer = ref();
const threejsContainer2 = ref();
let exampleClass: IsobandsExample;

const loadWasm = async () => {
  wasmInstance = await WASM();
};

const loadThreejs = async () => {
  if (wasmInstance === null) {
    await loadWasm();
  }
  const el = document.getElementById("threejsContainer");
  const el2 = document.getElementById("threejsContainer2");
  if (el && el2 && wasmInstance) {
    const meshes = [{ url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" }];
    for (const mesh of meshes) {
      const response = await fetch(mesh.url);
      const buffer = await response.arrayBuffer();
      wasmInstance.FS.writeFile(mesh.filename, new Int8Array(buffer));
    }
    exampleClass = new IsobandsExample(
      wasmInstance,
      meshes.map((m) => m.filename),
      el,
      el2,
      isDark.value,
    );
  }
};

const avgTime = ref("0");
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

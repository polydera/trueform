<script setup lang="ts">
import WASM from "@/examples/native";
import type { MainModule } from "@/examples/native";
import { BooleanExample } from "@/examples/BooleanExample";

let wasmInstance: MainModule | null = null;
const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");

const threejsContainer = ref();
const threejsContainer2 = ref();
let exampleClass: BooleanExample;

const loadWasm = async () => {
  wasmInstance = await WASM();
  console.log("Wasm loaded:", wasmInstance);
  console.log("Wasm loaded:", WASM);
};

const loadThreejs = async () => {
  if (wasmInstance === null) {
    await loadWasm();
  }
  let el = document.getElementById("threejsContainer");
  let el2 = document.getElementById("threejsContainer2");
  if (el && wasmInstance) {
    // Load data to wasm
    const meshes = [
      { url: "/stl/dragon-250k.stl", filename: "dragon-250k.stl" },
      { url: "/stl/Stanford_Bunny.stl", filename: "Stanford_Bunny.stl" },
    ];

    for (const mesh of meshes) {
      const response = await fetch(mesh.url);
      const buffer = await response.arrayBuffer();
      wasmInstance.FS.writeFile(mesh.filename, new Int8Array(buffer));
    }

    if (el2) {
      exampleClass = new BooleanExample(
        wasmInstance,
        meshes.map((m) => m.filename),
        el,
        el2,
        isDark.value,
      );
    }
  }
};

const avgTime = ref("0");
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
  <div class="flex flex-col w-full mt-(--ui-header-height)">
    <div class="flex flex-row flex-1 h-(calc(100vh - --ui-header-height)) relative">
      <div
        class="absolute left-3 top-3 z-10 max-w-md rounded-lg p-3 bg-neutral-100/10 shadow-lg backdrop-blur"
      >
        <p class="font-semibold text-lg mb-2">Boolean</p>
        <div class="flex flex-col gap-2">
          <div class="flex gap-3 items-center text-muted">
            <UIcon name="i-lucide-info" class="size-4 ml-1" />
            <p class="text-sm">Grab a mesh and move it to inspect intersection curve and difference mesh.</p>
          </div>
          <div class="flex gap-2 items-center text-muted">
            <UKbd variant="subtle">n</UKbd>
            <p class="text-sm">Randomize mesh orientation</p>
          </div>
        </div>
      </div>
      <div ref="threejsContainer" id="threejsContainer" class="h-full w-full m-0 p-0"></div>
      <div ref="threejsContainer2" id="threejsContainer2" class="h-full w-full m-0 p-0"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { MainModule } from "@/examples/native";
import { CollisionExample } from "@/examples/CollisionExample";

let wasmInstance: MainModule | null = null;
let wasmLoader: (() => Promise<MainModule>) | null = null;
const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");

const threejsContainer = ref();
let exampleClass: CollisionExample | null = null;
let CollisionExampleCtor: typeof import("@/examples/CollisionExample").CollisionExample | null = null;

const totalPolygons = ref("0");
const avgTime = ref("0");
const avgPickTime = ref("0");

const loadWasm = async () => {
  if (!wasmLoader) {
    const wasmModule = await import("@/examples/native");
    wasmLoader = wasmModule.default;
  }

  wasmInstance = await wasmLoader();
  console.log("Wasm loaded:", wasmInstance);
};

const loadThreejs = async () => {
  if (wasmInstance === null) {
    await loadWasm();
  }

  if (!CollisionExampleCtor) {
    const module = await import("@/examples/CollisionExample");
    CollisionExampleCtor = module.CollisionExample;
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

    exampleClass = new CollisionExample(
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
    avgTime.value = (exampleClass.getAverageTime() * 1000).toFixed(2);
    avgPickTime.value = (exampleClass.getAveragePickTime() * 1000).toFixed(2);
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
      <div class="absolute left-3 top-3 z-10 max-w-md rounded-lg p-3 bg-neutral-100/10 shadow-lg backdrop-blur">
        <p class="font-semibold text-lg mb-2">Collision</p>
        <div class="flex flex-col gap-2 text-sm">
          <div class="flex gap-3 items-center text-muted">
            <UIcon name="i-lucide-move-3d" class="size-4 ml-1" />
            <p>Grab and drag a mesh to test. Intersections highlight in real time.</p>
          </div>
          <div class="grid grid-cols-1 gap-1 text-muted">
            <p>Total polygons: {{ totalPolygons }}</p>
            <p>Pick: {{ avgPickTime }} us</p>
            <p>Collision: {{ avgTime }} us</p>
          </div>
        </div>
      </div>
      <div ref="threejsContainer" id="threejsContainer" class="h-full w-full m-0 p-0"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useWasmModule } from "@/composables/useWasmModule";
import { PositioningExample } from "@/examples/PositioningExample";

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");
const { loadWasmModule, preloadMeshes } = useWasmModule();

const threejsContainer = ref<HTMLElement | null>(null);
let exampleClass: PositioningExample | null = null;
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

  exampleClass = new PositioningExample(
    wasmInstance,
    meshes.map((m) => m.filename),
    el,
    isDark.value,
  );
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
      <div class="absolute left-3 top-3 z-10 max-w-md rounded-lg p-3 bg-neutral-100/10 shadow-lg backdrop-blur">
        <p class="font-semibold text-lg mb-2">Positioning</p>
        <div class="flex flex-col gap-2 text-sm text-muted">
          <div class="flex gap-3 items-center">
            <UIcon name="i-lucide-hand" class="size-4 ml-1" />
            <p>Drag a mesh away; release to see nearest neighbors snap back together.</p>
          </div>
        </div>
      </div>
      <div ref="threejsContainer" id="threejsContainer" class="h-full w-full flex-1 min-h-0 min-w-0"></div>
    </div>
  </div>
</template>

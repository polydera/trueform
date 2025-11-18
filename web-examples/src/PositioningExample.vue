<script setup lang="ts">
import WASM from './webAssembly/build/dist/native.js'
import { MainModule } from './webAssembly/build/dist/native.js'
import { PositioningExample } from "@/PositioningExample.js";
import { onMounted, ref } from "vue";

let wasmInstance: MainModule | null = null;

const threejsContainer = ref();
let exampleClass: PositioningExample;

const loadWasm = async () => {
  wasmInstance = await WASM();
};

const loadThreejs = async () => {
  if (wasmInstance === null) {
    await loadWasm();
  }
  const el = document.getElementById("threejsContainer");
  if (el && wasmInstance) {
    const path = "dragon-250k.stl";
    const path2 = "Stanford_Bunny.stl";

    const response = await fetch(path);
    const aBuff = await response.arrayBuffer();
    const intArr = new Int8Array(aBuff);
    wasmInstance.FS.writeFile(path, intArr);

    const response2 = await fetch(path2);
    const aBuff2 = await response2.arrayBuffer();
    const intArr2 = new Int8Array(aBuff2);
    wasmInstance.FS.writeFile(path2, intArr2);

    exampleClass = new PositioningExample(wasmInstance, [path, path2], el);
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
</script>

<template>
  <div style="display: flex; flex-direction: column; width: 100%; height: 100%;">
    <div style="display: flex; flex-direction: row; flex: 1 1 0">
      <div ref="threejsContainer" id="threejsContainer" style="height: 100%; width: 100%; margin: 0; padding: 0;"></div>
    </div>
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; margin: 10px;">
      <div style="display: flex; flex-direction: column;">
        <span style="white-space: pre-line; font-weight: bold">Drag one mesh away from the other.<br>Their nearest neighbors will be.<br>brought back together on release.</span>
      </div>
    </div>
  </div>
</template>

<style scoped></style>

<script setup lang="ts">
import WASM from './webAssembly/build/dist/native.js'
import { MainModule } from './webAssembly/build/dist/native.js'
import { BooleanExample } from "@/BooleanExample.js";
import {onMounted, ref} from "vue";

let wasmInstance: MainModule | null = null;

const threejsContainer = ref();
const threejsContainer2 = ref();
let exampleClass: BooleanExample;

const loadWasm = async () => {
  wasmInstance = await WASM()
  console.log("Wasm loaded:", wasmInstance)
  console.log("Wasm loaded:", WASM)
}

const loadThreejs = async () => {
  if(wasmInstance === null) {
    await loadWasm();
  }
  let el = document.getElementById("threejsContainer");
  let el2 = document.getElementById("threejsContainer2");
  if(el && wasmInstance) {
    // Load data to wasm
    // const path = "zan0.stl"
    const path = "dragon-250k.stl"
    const path2 = "Stanford_Bunny.stl"
    const response = await fetch(path);
    const aBuff = await response.arrayBuffer();
    const intArr = new Int8Array(aBuff);
    wasmInstance.FS.writeFile(path, intArr);
    const response2 = await fetch(path2);
    const aBuff2 = await response2.arrayBuffer();
    const intArr2 = new Int8Array(aBuff2);
    wasmInstance.FS.writeFile(path2, intArr2);

    if(el2) {
      exampleClass = new BooleanExample(wasmInstance, [path, path2], el, el2);
    }
  }
}

const avgTime = ref("0");
const getAvgTime = () => {
  if(exampleClass) {
    avgTime.value = exampleClass.getAverageTime().toFixed(2);
  }
  return 0;
}
setInterval(getAvgTime, 1000);

onMounted(() => {
  loadThreejs();
})
</script>

<template>

  <div style="display: flex; flex-direction: column; width: 100%; height: 100%">
    <div style="display: flex; flex-direction: row; flex: 1 1 0">
      <div ref="threejsContainer" id="threejsContainer" style="height: 100%; width: 100%; margin: 0; padding: 0;"></div>
      <div ref="threejsContainer2" id="threejsContainer2" style="height: 100%; width: 100%; margin: 0; padding: 0;"></div>
    </div>
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; margin: 10px">
      <div style="display: flex; flex-direction: column;">
        <span>Boolean time per frame: {{avgTime}} ms</span>
        <span>Press n to randomize mesh orientation.</span>
      </div>
      <div style="display: flex; flex-direction: column;">
        <span style="white-space: pre-line; font-weight: bold">Grab a mesh and move it.<br>See the intersection curve and the difference mesh.<br>Powered by trueform.</span>
      </div>
    </div>
  </div>
</template>

<style scoped></style>
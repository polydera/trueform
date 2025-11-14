<script setup lang="ts">
import WASM from './webAssembly/dist/native.js'
import { MainModule } from './webAssembly/dist/native.js'
import { IsobandsExample } from "@/IsobandsExample.js";
import {onMounted, ref} from "vue";

let wasmInstance: MainModule | null = null;

const threejsContainer = ref();
let exampleClass: IsobandsExample;

const loadWasm = async () => {
  wasmInstance = await WASM()
}

const loadThreejs = async () => {
  if(wasmInstance === null) {
    await loadWasm();
  }
  let el = document.getElementById("threejsContainer");
  let el2 = document.getElementById("threejsContainer2");
  if(el && el2 && wasmInstance) {
    const path = "dragon-250k.stl"

    const response = await fetch(path);
    const aBuff = await response.arrayBuffer();
    const intArr = new Int8Array(aBuff);
    wasmInstance.FS.writeFile(path, intArr);

    exampleClass = new IsobandsExample(wasmInstance, [path], el, el2);
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
  <div style="display: flex; flex-direction: column; width: 100%; height: 100%;">
    <div style="display: flex; flex-direction: row; justify-content: space-evenly;">
      <div style="display: flex; flex-direction: column;">
        <span>Isobands time per scroll: {{avgTime}} mcs</span>
        <span>Press n to randomize the plane.</span>
      </div>
      <div style="display: flex; flex-direction: column;">
        <span style="white-space: pre-line; font-weight: bold">Hold shift and scroll.<br>Intersection curve with plane will move.<br>Powered by trueform.</span>
      </div>
    </div>
    <div style="display: flex; flex-direction: row; flex: 1 1 0; margin-top: 20px;">
      <div ref="threejsContainer" id="threejsContainer" style="height: 100%; width: 100%; margin: 0; padding: 0;"></div>
      <div ref="threejsContainer2" id="threejsContainer2" style="height: 100%; width: 100%; margin: 0; padding: 0;"></div>
    </div>
  </div>
</template>

<style scoped></style>
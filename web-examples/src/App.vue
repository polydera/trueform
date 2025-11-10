<script setup lang="ts">
import WASM from './webAssembly/dist/native.js'
import { MainModule } from './webAssembly/dist/native.js'
import { TestClassThreejs } from "@/TestThreejs.js";
import {onMounted, ref} from "vue";

let wasmInstance: MainModule | null = null;

const threejsContainer = ref();
const threejsContainer2 = ref();
let testObjThreejs: TestClassThreejs;

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
    // const path = "dragon-250k.stl"
    const path = "Stanford_Bunny.stl"
    const response = await fetch(path);
    const aBuff = await response.arrayBuffer();
    const intArr = new Int8Array(aBuff);
    wasmInstance.FS.writeFile(path, intArr);

    testObjThreejs = new TestClassThreejs(wasmInstance, path, el, el2 ?? undefined);
  }
}

const avgTime = ref("0");
const getAvgTime = () => {
  if(testObjThreejs) {
    avgTime.value = testObjThreejs.getAverageBooleanTime().toFixed(2);
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
    <div style="display: flex; justify-content: center">Average Boolean time: {{avgTime}} ms</div>
    <div style="display: flex; flex-direction: row; width: 100%; height: 100%;">
      <div ref="threejsContainer" id="threejsContainer" style="height: 100%; width: 100%; margin: 0; padding: 0;"></div>
      <div ref="threejsContainer2" id="threejsContainer2" style="height: 100%; width: 100%; margin: 0; padding: 0;"></div>
    </div>
  </div>
</template>

<style scoped></style>

<script setup lang="ts">
import WASM from './webAssembly/dist/native.js'
import { MainModule } from './webAssembly/dist/native.js'
import { TestClassThreejs } from "@/TestThreejs.js";
import {onMounted, ref} from "vue";

let wasmInstance: MainModule | null = null;

const threejsContainer = ref();
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
  console.log("el: ", el)
  if(el && wasmInstance) {
    // Load data to wasm
    const path = "zan0.stl"
    const response = await fetch(path);
    const aBuff = await response.arrayBuffer();
    const intArr = new Int8Array(aBuff);
    wasmInstance.FS.writeFile(path, intArr);
    console.log("File written to wasm:", intArr)

    testObjThreejs = new TestClassThreejs(wasmInstance, el);
  }
}

onMounted(() => {
  loadThreejs();
})
</script>

<template>
    <div ref="threejsContainer" id="threejsContainer" style="height: 100%; width: 100%; margin: 0; padding: 0;"></div>
</template>

<style scoped></style>

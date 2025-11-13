<template>
  <div style="display: flex; flex-direction: column; width: 100%; height: 100%; overflow: hidden;">
    <div style="display: flex">
      <div v-for="route in exampleRoutes" :key="route.path" style="padding: 10px">
        <button @click="router.push(route.path)" style="border-radius: 16px; background: #7799ee; height: 50px">{{ route.label }}</button>
      </div>
    </div>
    <RouterView></RouterView>
  </div>
</template>

<script lang="ts" setup>
import { useRouter } from "vue-router";
import { exampleRoutes } from "./Routes";
const router = useRouter();
</script>

<style scoped></style>

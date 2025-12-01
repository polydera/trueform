<script setup lang="ts">
import { useIntervalFn, useElementHover } from "@vueuse/core";

const numCharts = 5;
const currentChart = ref(0);
const carouselRef = ref<HTMLElement | null>(null);

const nextChart = () => {
  currentChart.value = (currentChart.value + 1) % numCharts;
};

const isHovered = useElementHover(carouselRef);
const { pause, resume } = useIntervalFn(nextChart, 5000);

watch(isHovered, (hovered) => {
  if (hovered) {
    pause();
  } else {
    resume();
  }
});
</script>

<template>
  <div ref="carouselRef" class="w-full overflow-hidden relative">
    <TransitionGroup name="slide" tag="div" class="relative">
      <div v-if="currentChart === 0" key="chart-0" class="w-full">
        <ChartsCutBooleanSpeedup />
      </div>
      <div v-if="currentChart === 1" key="chart-1" class="w-full">
        <ChartsSpatialPolygonsBuildAabbSpeedup />
      </div>
      <div v-if="currentChart === 2" key="chart-2" class="w-full">
        <ChartsIntersectIsocontoursSpeedup />
      </div>
      <div v-if="currentChart === 3" key="chart-3" class="w-full">
        <ChartsSpatialMeshMeshDistanceObbrssSpeedup />
      </div>
      <div v-if="currentChart === 4" key="chart-4" class="w-full">
        <ChartsSpatialKnnSpeedup />
      </div>
    </TransitionGroup>
  </div>
</template>

<script setup lang="ts">
import {
  VisXYContainer,
  VisLine,
  VisAxis,
  VisCrosshair,
  VisTooltip,
  VisAnnotations,
} from "@unovis/vue";
import rawData from "../../../benchmarks/point_cloud_knn.json";

// Filter for k=10 (hardest query) to show scaling with point count
const data = rawData.filter((d: any) => d.k === 10);

const x = (d: any) => d.points;
const y = [
  (d: any) => d.tf_obbrss * 1000,  // Convert to µs
  (d: any) => d.nanoflann * 1000,
];
const color = (_: any, i: number) => ["#00d5be", "#a82d12"][i];

const round = (n: number) => Math.round(n * 100) / 100;
const template = (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${numKM(d.points)} points, k=${d.k}</div>
    <div><span class="text-primary font-bold">TF OBBRSS:</span> ${round(d.tf_obbrss * 1000)} µs</div>
    <div><span class="text-[#a82d12]">nanoflann:</span> ${round(d.nanoflann * 1000)} µs</div>
  </div>`;

const lastPoint = data[data.length - 1];
const maxY = Math.max(...data.flatMap((d: any) => [d.tf_obbrss * 1000, d.nanoflann * 1000]));
const yDomain: [number, number] = [0, maxY * 1.1]; // Start at 0 with 10% padding
const tfYPercent = 100 - ((lastPoint.tf_obbrss * 1000) / yDomain[1] * 100);

const annotations = computed(() => [
  {
    x: "75%",
    y: `${Math.max(tfYPercent - 10, 5)}%`,
    content: {
      text: `${round(lastPoint.tf_obbrss * 1000)} µs`,
      fontSize: 16,
      fontWeight: 700,
      color: "#00d5be",
    },
    subject: {
      x: "100%",
      y: `${tfYPercent}%`,
      connectorLineStrokeDasharray: "2 2",
      radius: 3,
    },
  },
]);
</script>
<template>
  <div class="w-full unovis flex flex-col gap-2.5 items-center justify-center">
    <h2 class="text-xl font-medium text-center">
      k-Nearest Neighbor Query (k=10)
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-primary rounded"></div>
        <NuxtImg src="/tf.png" class="h-4 w-auto shrink-0" />
      </div>
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#a82d12] rounded"></div>
        <NuxtImg src="/img/nanoflann_logo.png" class="h-4 w-auto shrink-0" />
      </div>
    </div>
    <VisXYContainer :data="data" :y-domain="yDomain">
      <VisLine :x="x" :y="y" :color="color" :duration="1200" />
      <VisTooltip />
      <VisAxis type="x" label="Number of Points" :tickFormat="(value: number) => numKM(value)" />
      <VisAxis type="y" label="Time [µs]" />
      <VisCrosshair :template="template" :color="color" />
      <!-- @vue-expect-error -->
      <VisAnnotations :items="annotations" />
    </VisXYContainer>
  </div>
</template>

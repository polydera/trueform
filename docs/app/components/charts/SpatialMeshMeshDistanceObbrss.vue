<script setup lang="ts">
import {
  VisXYContainer,
  VisLine,
  VisAxis,
  VisCrosshair,
  VisTooltip,
  VisAnnotations,
} from "@unovis/vue";
import data from "../../../benchmarks/mesh_mesh_closest_point.json";

const x = (d: any) => d.polygons;
const y = [
  (d: any) => d.tf_obbrss,
  (d: any) => d.fcl_obbrss,
  (d: any) => d.coal_obbrss,
];
const color = (_: any, i: number) => ["#00d5be", "#9b59b6", "#e74c3c"][i];

const round = (n: number) => Math.round(n * 1e3) / 1e3;
const template = (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">2 × ${numKM(d.polygons)} polygons</div>
    <div><span class="text-primary font-bold">TF OBBRSS:</span> ${round(d.tf_obbrss)} ms</div>
    <div><span class="text-[#9b59b6]">FCL OBBRSS:</span> ${round(d.fcl_obbrss)} ms</div>
    <div><span class="text-[#e74c3c]">Coal OBBRSS:</span> ${round(d.coal_obbrss)} ms</div>
  </div>`;

const lastPoint = data[data.length - 1];
const maxY = Math.max(...data.flatMap((d: any) => [d.tf_obbrss, d.fcl_obbrss, d.coal_obbrss]));
const yDomain: [number, number] = [0, maxY * 1.1]; // Start at 0 with 10% padding
const tfYPercent = 100 - (lastPoint.tf_obbrss / yDomain[1] * 100);

const annotations = computed(() => [
  {
    x: "75%",
    y: `${Math.max(tfYPercent - 10, 5)}%`,
    content: {
      text: `${round(lastPoint.tf_obbrss)} ms`,
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
      Mesh-to-Mesh Distance (OBBRSS)
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-primary rounded"></div>
        <NuxtImg src="/tf.png" class="h-4 w-auto shrink-0" />
      </div>
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#9b59b6] rounded"></div>
        <span class="text-sm">FCL</span>
      </div>
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#e74c3c] rounded"></div>
        <span class="text-sm">Coal</span>
      </div>
    </div>
    <VisXYContainer :data="data" :y-domain="yDomain">
      <VisLine :x="x" :y="y" :color="color" :duration="1200" />
      <VisTooltip />
      <VisAxis type="x" label="Number of Polygons" :tickFormat="(value: number) => `2×${numKM(value)}`" />
      <VisAxis type="y" label="Time [ms]" />
      <VisCrosshair :template="template" :color="color" />
      <!-- @vue-expect-error -->
      <VisAnnotations :items="annotations" />
    </VisXYContainer>
  </div>
</template>

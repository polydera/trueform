<script setup lang="ts">
import {
  VisXYContainer,
  VisLine,
  VisAxis,
  VisCrosshair,
  VisTooltip,
  VisAnnotations,
} from "@unovis/vue";
import data from "../../../benchmarks/polygons_build_tree.json";

const x = (d: any) => d.polygons;
const y = [
  (d: any) => d.tf_obb,
  (d: any) => d.fcl_obb,
  (d: any) => d.coal_obb,
];
const color = (_: any, i: number) => ["#00d5be", "#9b59b6", "#e74c3c"][i];

const round = (n: number) => Math.round(n * 1e2) / 1e2;
const template = (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${numKM(d.polygons)} polygons</div>
    <div><span class="text-primary font-bold">TF OBB:</span> ${round(d.tf_obb)} ms</div>
    <div><span class="text-[#9b59b6]">FCL OBB:</span> ${round(d.fcl_obb)} ms</div>
    <div><span class="text-[#e74c3c]">Coal OBB:</span> ${round(d.coal_obb)} ms</div>
  </div>`;

const lastPoint = data[data.length - 1];
const maxY = Math.max(...data.flatMap((d: any) => [d.tf_obb, d.fcl_obb, d.coal_obb]));
const tfYPercent = 100 - (lastPoint.tf_obb / maxY * 100);

const annotations = computed(() => [
  {
    x: "75%",
    y: `${Math.max(tfYPercent - 10, 5)}%`,
    content: {
      text: `${round(lastPoint.tf_obb)} ms`,
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
      BVH Construction (OBB)
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
    <VisXYContainer :data="data">
      <VisLine :x="x" :y="y" :color="color" :duration="1200" />
      <VisTooltip />
      <VisAxis type="x" label="Number of Polygons" :tickFormat="(value: number) => numKM(value)" />
      <VisAxis type="y" label="Time [ms]" />
      <VisCrosshair :template="template" :color="color" />
      <!-- @vue-expect-error -->
      <VisAnnotations :items="annotations" />
    </VisXYContainer>
  </div>
</template>

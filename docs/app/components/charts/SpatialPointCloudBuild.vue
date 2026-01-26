<script setup lang="ts">
import {
  VisXYContainer,
  VisLine,
  VisAxis,
  VisCrosshair,
  VisTooltip,
  VisAnnotations,
} from "@unovis/vue";
import data from "../../../benchmarks/point_cloud_build_tree.json";

const x = (d: any) => d.points;
const y = [
  (d: any) => d.tf_aabb,
  (d: any) => d.tf_obb,
  (d: any) => d.tf_obbrss,
  (d: any) => d.nanoflann,
];
const color = (_: any, i: number) => ["#00d5be", "#00a896", "#007d6e", "#a82d12"][i];

const round = (n: number) => Math.round(n * 1e2) / 1e2;
const template = (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${numKM(d.points)} points</div>
    <div><span class="text-primary font-bold">TF AABB:</span> ${round(d.tf_aabb)} ms</div>
    <div><span class="text-[#00a896]">TF OBB:</span> ${round(d.tf_obb)} ms</div>
    <div><span class="text-[#007d6e]">TF OBBRSS:</span> ${round(d.tf_obbrss)} ms</div>
    <div><span class="text-[#a82d12]">nanoflann:</span> ${round(d.nanoflann)} ms</div>
  </div>`;

const lastPoint = data[data.length - 1];
const maxY = Math.max(...data.flatMap((d: any) => [d.tf_aabb, d.tf_obb, d.tf_obbrss, d.nanoflann]));
const tfYPercent = 100 - (lastPoint.tf_aabb / maxY * 100);

const annotations = computed(() => [
  {
    x: "75%",
    y: `${Math.max(tfYPercent - 10, 5)}%`,
    content: {
      text: `${round(lastPoint.tf_aabb)} ms`,
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
      Point Cloud Tree Construction
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-primary rounded"></div>
        <span class="text-sm">TF AABB</span>
      </div>
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#00a896] rounded"></div>
        <span class="text-sm">TF OBB</span>
      </div>
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#007d6e] rounded"></div>
        <span class="text-sm">TF OBBRSS</span>
      </div>
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#a82d12] rounded"></div>
        <NuxtImg src="/img/nanoflann_logo.png" class="h-4 w-auto shrink-0" />
      </div>
    </div>
    <VisXYContainer :data="data">
      <VisLine :x="x" :y="y" :color="color" :duration="1200" />
      <VisTooltip />
      <VisAxis type="x" label="Number of Points" :tickFormat="(value: number) => numKM(value)" />
      <VisAxis type="y" label="Time [ms]" />
      <VisCrosshair :template="template" :color="color" />
      <!-- @vue-expect-error -->
      <VisAnnotations :items="annotations" />
    </VisXYContainer>
  </div>
</template>

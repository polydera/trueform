<script setup lang="ts">
import {
  VisXYContainer,
  VisLine,
  VisAxis,
  VisCrosshair,
  VisTooltip,
  VisAnnotations,
} from "@unovis/vue";
import rawData from "../../../benchmarks/embedded_isocurves.json";

// Filter for n_cuts=16 to show scaling with mesh size
const data = rawData.filter((d: any) => d.n_cuts === 16);

const x = (d: any) => d.polygons;
const y = [
  (d: any) => d.tf,
  (d: any) => d.vtk,
];
const color = (_: any, i: number) => ["#00d5be", "#00529b"][i];

const round = (n: number) => Math.round(n * 100) / 100;
const template = (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${numKM(d.polygons)} polygons, ${d.n_cuts} cuts</div>
    <div><span class="text-primary font-bold">TrueForm:</span> ${round(d.tf)} ms</div>
    <div><span class="text-[#00529b]">VTK:</span> ${round(d.vtk)} ms</div>
  </div>`;

const lastPoint = data[data.length - 1];
const maxY = Math.max(...data.flatMap((d: any) => [d.tf, d.vtk]));
const yDomain: [number, number] = [0, maxY * 1.1];
const tfYPercent = 100 - (lastPoint.tf / yDomain[1] * 100);

const annotations = computed(() => [
  {
    x: "75%",
    y: `${Math.max(tfYPercent - 10, 5)}%`,
    content: {
      text: `${round(lastPoint.tf)} ms`,
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
      Embedded Isocurves (16 cuts)
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-primary rounded"></div>
        <NuxtImg src="/tf.png" class="h-4 w-auto shrink-0" />
      </div>
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#00529b] rounded"></div>
        <NuxtImg src="/img/vtk_logo.png" class="h-4 w-auto shrink-0" />
      </div>
    </div>
    <VisXYContainer :data="data" :y-domain="yDomain">
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

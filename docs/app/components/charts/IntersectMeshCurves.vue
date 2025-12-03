<script setup lang="ts">
import {
  VisXYContainer,
  VisLine,
  VisAxis,
  VisCrosshair,
  VisTooltip,
  VisAnnotations,
} from "@unovis/vue";
import data from "../../../benchmarks/mesh_mesh_curves.json";

const x = (d: any) => d.polygons;
const y = [
  (d: any) => d.tf,
  (d: any) => d.cgal,
];
const color = (_: any, i: number) => ["#00d5be", "#fdff4e"][i];

const round = (n: number) => Math.round(n * 1e2) / 1e2;
const template = (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">2 × ${numKM(d.polygons)} polygons</div>
    <div><span class="text-primary font-bold">TrueForm:</span> ${round(d.tf)} ms</div>
    <div><span class="text-[#fdff4e]">CGAL:</span> ${round(d.cgal)} ms</div>
  </div>`;

const lastPoint = data[data.length - 1];
const maxY = Math.max(...data.flatMap((d: any) => [d.tf, d.cgal]));
const tfYPercent = 100 - (lastPoint.tf / maxY * 100);

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
      Mesh-Mesh Intersection Curves
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-primary rounded"></div>
        <NuxtImg src="/tf.png" class="h-4 w-auto shrink-0" />
      </div>
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#fdff4e] rounded"></div>
        <img src="/img/cgal_logo.png" class="h-4 w-auto shrink-0" alt="CGAL" />
      </div>
    </div>
    <VisXYContainer :data="data">
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

<script setup lang="ts">
import {
  VisXYContainer,
  VisLine,
  VisAxis,
  VisCrosshair,
  VisTooltip,
} from "@unovis/vue";
import data from "../../../benchmarks/isotropic_remeshing_multiplier.json";

const x = (d: any) => d.multiplier;
const y = [
  (d: any) => d.tf,
  (d: any) => d.cgal,
];
const color = (_: any, i: number) => ["#00d5be", "#fdff4e"][i];

const round = (n: number) => Math.round(n * 1e2) / 1e2;
const template = (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${d.multiplier}× mel</div>
    <div><span class="text-primary font-bold">TrueForm:</span> ${round(d.tf)} ms</div>
    <div><span class="text-[#fdff4e]">CGAL:</span> ${round(d.cgal)} ms</div>
    <div class="text-neutral-400 text-sm">${round(d.cgal / d.tf)}× speedup</div>
  </div>`;
</script>
<template>
  <div class="w-full unovis flex flex-col gap-2.5 items-center justify-center">
    <h2 class="text-xl font-medium text-center">
      Isotropic Remeshing by Multiplier (1M polygons)
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
      <VisAxis type="x" label="Target Length Multiplier (× mel)" :tickFormat="(value: number) => value + '×'" />
      <VisAxis type="y" label="Time [ms]" />
      <VisCrosshair :template="template" :color="color" />
    </VisXYContainer>
  </div>
</template>

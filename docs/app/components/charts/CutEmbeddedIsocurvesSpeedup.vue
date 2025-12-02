<script setup lang="ts">
import { VisXYContainer, VisGroupedBar, VisAxis, VisTooltip } from "@unovis/vue";
import { GroupedBar } from "@unovis/ts";
import rawData from "../../../benchmarks/embedded_isocurves.json";

// Filter for n_cuts=16
const data = rawData.filter((d: any) => d.n_cuts === 16);

const x = (_: any, i: number) => i;
const y = [(d: any) => d.vtk / d.tf];
const color = (_: any, i: number) => ["#00529b"][i];

const round = (n: number) => Math.round(n);
const triggers = {
  [GroupedBar.selectors.bar]: (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${numKM(d.polygons)} polygons, ${d.n_cuts} cuts</div>
    <div><span class="text-[#00529b]">vs VTK:</span> ${round(d.vtk / d.tf)}×</div>
  </div>`,
};
</script>
<template>
  <div class="w-full unovis flex flex-col gap-2.5 items-center justify-center">
    <h2 class="text-xl font-medium text-center">
      Embedded Isocurves Speedup (16 cuts)
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#00529b] rounded"></div>
        <span class="text-sm">vs</span>
        <NuxtImg src="/img/vtk_logo.png" class="h-4 w-auto shrink-0" />
      </div>
    </div>
    <VisXYContainer>
      <VisGroupedBar :data="data" :x="x" :y="y" :color="color" :duration="300" />
      <VisTooltip :triggers="triggers" />
      <VisAxis
        type="x"
        label="Number of Polygons"
        :tickFormat="(value: number) => numKM(data[value]?.polygons) || ''"
        :numTicks="data?.length || 0"
      />
      <VisAxis type="y" label="Speedup Factor" />
    </VisXYContainer>
  </div>
</template>

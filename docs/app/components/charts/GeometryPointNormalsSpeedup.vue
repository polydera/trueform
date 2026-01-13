<script setup lang="ts">
import { VisXYContainer, VisGroupedBar, VisAxis, VisTooltip } from "@unovis/vue";
import { GroupedBar } from "@unovis/ts";
import data from "../../../benchmarks/point_normals.json";

const x = (_: any, i: number) => i;
const y = [
  (d: any) => d.igl / d.tf,
  (d: any) => d.vtk / d.tf,
];
const color = (_: any, i: number) => ["#ff6b6b", "#00529b"][i];

const round = (n: number) => Math.round(n * 10) / 10;
const triggers = {
  [GroupedBar.selectors.bar]: (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${numKM(d.polygons)} polygons</div>
    <div><span class="text-[#ff6b6b]">vs libigl:</span> ${round(d.igl / d.tf)}×</div>
    <div><span class="text-[#00529b]">vs VTK:</span> ${round(d.vtk / d.tf)}×</div>
  </div>`,
};
</script>
<template>
  <div class="w-full unovis flex flex-col gap-2.5 items-center justify-center">
    <h2 class="text-xl font-medium text-center">
      Point Normals (Speedup)
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#ff6b6b] rounded"></div>
        <span class="text-sm">vs libigl</span>
      </div>
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

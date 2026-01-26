<script setup lang="ts">
import { VisXYContainer, VisGroupedBar, VisAxis, VisTooltip } from "@unovis/vue";
import { GroupedBar } from "@unovis/ts";
import data from "../../../benchmarks/point_cloud_build_tree.json";

const x = (_: any, i: number) => i;
const y = [(d: any) => d.nanoflann / d.tf_aabb];
const color = (_: any, i: number) => ["#a82d12"][i];

const round = (n: number) => Math.round(n * 10) / 10;
const triggers = {
  [GroupedBar.selectors.bar]: (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${numKM(d.points)} points</div>
    <div><span class="text-[#a82d12]">vs nanoflann:</span> ${round(d.nanoflann / d.tf_aabb)}×</div>
  </div>`,
};
</script>
<template>
  <div class="w-full unovis flex flex-col gap-2.5 items-center justify-center">
    <h2 class="text-xl font-medium text-center">
      Point Cloud Tree Construction (Speedup)
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#a82d12] rounded"></div>
        <span class="text-sm">vs</span>
        <NuxtImg src="/img/nanoflann_logo.png" class="h-4 w-auto shrink-0" />
      </div>
    </div>
    <VisXYContainer>
      <VisGroupedBar :data="data" :x="x" :y="y" :color="color" :duration="300" />
      <VisTooltip :triggers="triggers" />
      <VisAxis
        type="x"
        label="Number of Points"
        :tickFormat="(value: number) => numKM(data[value]?.points) || ''"
        :numTicks="data?.length || 0"
      />
      <VisAxis type="y" label="Speedup Factor" />
    </VisXYContainer>
  </div>
</template>

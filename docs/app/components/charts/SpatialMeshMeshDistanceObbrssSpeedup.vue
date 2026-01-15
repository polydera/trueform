<script setup lang="ts">
import { VisXYContainer, VisGroupedBar, VisAxis, VisTooltip } from "@unovis/vue";
import { GroupedBar } from "@unovis/ts";
import data from "../../../benchmarks/mesh_mesh_closest_point.json";

const x = (_: any, i: number) => i;
const y = [
  (d: any) => d.fcl_obbrss / d.tf_obbrss,
  (d: any) => d.coal_obbrss / d.tf_obbrss,
];
const color = (_: any, i: number) => ["#9b59b6", "#e74c3c"][i];

const round = (n: number) => Math.round(n * 10) / 10;
const triggers = {
  [GroupedBar.selectors.bar]: (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">2 × ${numKM(d.polygons)} polygons</div>
    <div><span class="text-[#9b59b6]">vs FCL:</span> ${round(d.fcl_obbrss / d.tf_obbrss)}×</div>
    <div><span class="text-[#e74c3c]">vs Coal:</span> ${round(d.coal_obbrss / d.tf_obbrss)}×</div>
  </div>`,
};
</script>
<template>
  <div class="w-full unovis flex flex-col gap-2.5 items-center justify-center">
    <h2 class="text-xl font-medium text-center">
      Mesh-to-Mesh Distance OBBRSS (Speedup)
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#9b59b6] rounded"></div>
        <span class="text-sm">vs FCL</span>
      </div>
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#e74c3c] rounded"></div>
        <span class="text-sm">vs Coal</span>
      </div>
    </div>
    <VisXYContainer>
      <VisGroupedBar :data="data" :x="x" :y="y" :color="color" :duration="300" />
      <VisTooltip :triggers="triggers" />
      <VisAxis
        type="x"
        label="Number of Polygons"
        :tickFormat="(value: number) => data[value]?.polygons ? `2×${numKM(data[value].polygons)}` : ''"
        :numTicks="data?.length || 0"
      />
      <VisAxis type="y" label="Speedup Factor" />
    </VisXYContainer>
  </div>
</template>

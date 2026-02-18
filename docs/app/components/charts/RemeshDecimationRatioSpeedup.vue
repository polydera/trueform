<script setup lang="ts">
import { VisXYContainer, VisGroupedBar, VisAxis, VisTooltip } from "@unovis/vue";
import { GroupedBar } from "@unovis/ts";
import data from "../../../benchmarks/decimation_ratio.json";

const x = (_: any, i: number) => i;
const y = [
  (d: any) => d.cgal / d.tf,
];
const color = (_: any, i: number) => ["#fdff4e"][i];

const round = (n: number) => Math.round(n * 10) / 10;
const triggers = {
  [GroupedBar.selectors.bar]: (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">Ratio: ${d.ratio}</div>
    <div><span class="text-[#fdff4e]">vs CGAL:</span> ${round(d.cgal / d.tf)}×</div>
  </div>`,
};
</script>
<template>
  <div class="w-full unovis flex flex-col gap-2.5 items-center justify-center">
    <h2 class="text-xl font-medium text-center">
      Decimation by Ratio (Speedup)
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#fdff4e] rounded"></div>
        <span class="text-sm">vs</span>
        <img src="/img/cgal_logo.png" class="h-4 w-auto shrink-0" alt="CGAL" />
      </div>
    </div>
    <VisXYContainer>
      <VisGroupedBar :data="data" :x="x" :y="y" :color="color" :duration="300" />
      <VisTooltip :triggers="triggers" />
      <VisAxis
        type="x"
        label="Target Ratio"
        :tickFormat="(value: number) => data[value]?.ratio?.toFixed(1) || ''"
        :numTicks="data?.length || 0"
      />
      <VisAxis type="y" label="Speedup Factor" />
    </VisXYContainer>
  </div>
</template>

<script setup lang="ts">
import {
  VisXYContainer,
  VisLine,
  VisAxis,
  VisCrosshair,
  VisTooltip,
} from "@unovis/vue";
import rawData from "../../../benchmarks/mod_tree_update.json";

const polygonCounts = Object.keys(rawData[0]).filter(k => k !== "dirty_pct").map(Number).sort((a, b) => a - b);

// Compute efficiency: dirty_pct / update_pct (ideal = 1.0)
const data = rawData.map((d: any) => {
  const row: any = { dirty_pct: d.dirty_pct };
  for (const p of polygonCounts) {
    row[p] = d.dirty_pct / d[p];
  }
  return row;
});

const x = (d: any) => d.dirty_pct;
const y = polygonCounts.map(p => (d: any) => d[p]);
const colors = ["#00d5be", "#00c4ad", "#00a896", "#007d6e", "#005a4e", "#003d35"];
const color = (_: any, i: number) => colors[i];

const round = (n: number) => Math.round(n * 100) / 100;
const template = (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${d.dirty_pct}% dirty</div>
    ${polygonCounts.map((p, i) => `<div><span style="color: ${colors[i]}; font-weight: bold;">${numKM(p)}:</span> ${round(d[p])}</div>`).join('')}
  </div>`;
</script>
<template>
  <div class="w-full unovis flex flex-col gap-2.5 items-center justify-center">
    <h2 class="text-xl font-medium text-center">
      Update Efficiency (1.0 = ideal)
    </h2>
    <div class="flex gap-4 items-center justify-center flex-wrap">
      <div v-for="(p, i) in polygonCounts" :key="p" class="flex gap-1.5 items-center">
        <div class="size-3 rounded" :style="{ backgroundColor: colors[i] }"></div>
        <span class="text-sm">{{ numKM(p) }}</span>
        <UIcon name="i-lucide-triangle" class="size-3" />
      </div>
    </div>
    <VisXYContainer :data="data" :yDomain="[0, 1.1]">
      <VisLine :x="x" :y="y" :color="color" :duration="1200" />
      <VisTooltip />
      <VisAxis type="x" label="Dirty Primitives [%]" :tickFormat="(value: number) => `${value}%`" />
      <VisAxis type="y" label="Efficiency" :tickFormat="(value: number) => value > 1 ? '' : value.toString()" />
      <VisCrosshair :template="template" :color="color" />
    </VisXYContainer>
  </div>
</template>

<script setup lang="ts">
import { VisXYContainer, VisGroupedBar, VisAxis, VisCrosshair, VisTooltip } from "@unovis/vue";
import { FitMode, GroupedBar } from "@unovis/ts";
import mesh_intersection_curves from "../../../benchmarks/vtk_feature_edges_non_manifold.json";

const data = mesh_intersection_curves;
const len = ref(1);

// const dataS = computed(() => data.map((d, i) => (i < len.value ? d : { ...d, vtk: 0, cgal: 0 })));
const dataS = computed(() => data.slice(0, len.value));

onMounted(() => {
  const interval = setInterval(() => {
    if (len.value < data.length) {
      len.value += 1;
    } else {
      clearInterval(interval);
    }
  }, 300);
});

const x = (d: any, i: number) => i;
const y = (d: any) => d.speedup;
const color = (_: any, i: number) => ["#00529b"][i];
const round = (n: number) => Math.round(n * 1e2) / 1e2;
const triggers = {
  [GroupedBar.selectors.bar]: (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${d.triangles} triangles</div>
    <div><span class="text-[#00529b]">VTK:</span> ${round(d.speedup)}x</div>
  </div>`,
};
</script>
<template>
  <div class="w-full unovis flex flex-col gap-2.5 items-center justify-center">
    <h2 class="text-xl font-medium text-center">
      Feature Edges (Non-Manifold) :<br />
      Speed-up of <span class="text-primary font-bold">trueform</span> vs VTK
    </h2>
    <div class="flex gap-4 items-center justify-center">
      <div class="flex gap-1.5 items-center">
        <div class="size-3 bg-[#00529b] rounded"></div>
        <NuxtImg src="/img/vtk_logo.png" class="h-4 w-auto shrink-0" />
      </div>
    </div>
    <VisXYContainer :y-domain="[0, 30]">
      <VisGroupedBar :data="dataS" :x="x" :y="y" :color="color" :duration="300" />
      <VisTooltip :triggers="triggers" />
      <VisAxis
        type="x"
        label="Number of Triangles"
        :tickFormat="(value: number) => dataS[value]?.triangles || ''"
        :numTicks="dataS?.length || 0"
      />
      <VisAxis
        type="y"
        :tickFormat="(value: number) => `${value} x`"
        label="Speed-up"
        :tick-text-fit-mode="FitMode.Trim"
      />
    </VisXYContainer>
  </div>
</template>

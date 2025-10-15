<script setup lang="ts">
import {
  VisXYContainer,
  VisLine,
  VisAxis,
  VisCrosshair,
  VisTooltip,
  VisAnnotations,
} from "@unovis/vue";
import planar_arrangements from "../../../benchmarks/planar_arrangements.json";

const data = planar_arrangements;

const len = ref(1);
const dataS = computed(() => data);

onMounted(() => {
  const interval = setInterval(() => {
    if (len.value < data.length) {
      len.value += 5;
    } else {
      clearInterval(interval);
    }
  }, 600);
});

const x = (d: any) => d.edges;
const y = [(d: any) => d.trueform_ms, (d: any) => d.cgal_ms];
const color = (_: any, i: number) => ["#00d5be", "#fdff4e"][i];

const colorMode = useColorMode();
const isDark = computed(() => colorMode.value === "dark");

const round = (n: number) => Math.round(n * 1e2) / 1e2;
const template = (d: any) => `<div class="flex flex-col gap-0.5">
    <div class="font-medium text-lg">${d.edges} edges</div>
    <div><span class="text-primary font-bold">trueform:</span> ${round(d.trueform_ms)} ms</div>
    <div><span class="text-[#fdff4e]">CGAL:</span> ${round(d.cgal_ms)} ms</div>
  </div>`;

const annotations = computed(() => [
  {
    x: "85%",
    y: "90%",
    // width: 100,
    content: {
      text: `${round(data[data.length - 1]!.trueform_ms)} ms`,
      fontSize: 20,
      fontWeight: 700,
      color: isDark.value ? "#ffffff" : "#000000",
    },
    subject: {
      x: "100%",
      y: "99%",
      connectorLineStrokeDasharray: "2 2",
      radius: 3,
    },
    textAlign: "center",
    verticalAlign: "bottom",
  },
]);
</script>
<template>
  <UContainer>
    <div class="w-full unovis my-5 flex flex-col gap-2.5 items-center justify-center">
      <h2 class="text-xl font-medium text-center">
        Planar Arrangements:<br />
        Comparison of <span class="text-primary font-bold">trueform</span> vs CGAL
      </h2>
      <div class="flex gap-4 items-center justify-center">
        <div class="flex gap-1.5 items-center">
          <div class="size-3 bg-primary rounded"></div>
          <NuxtImg src="/tf.png" class="h-4 w-auto shrink-0" />
        </div>
        <div class="flex gap-1.5 items-center">
          <div class="size-3 bg-[#fdff4e] rounded"></div>
          <NuxtImg src="/img/cgal_logo.png" class="h-4 w-auto shrink-0" />
        </div>
      </div>
      <VisXYContainer :data="dataS" :x-domain="[0, 40000]">
        <VisLine :x="x" :y="y" :color="color" :duration="1200" />
        <VisTooltip />
        <VisAxis type="x" label="Number of Edges in the Arrangement" />
        <VisAxis type="y" label="Time [ms]" />
        <VisCrosshair :template="template" :color="color" />
        <!-- @vue-expect-error -->
        <VisAnnotations :items="annotations" />
      </VisXYContainer>
    </div>
  </UContainer>
</template>

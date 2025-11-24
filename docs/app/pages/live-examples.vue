<script setup lang="ts">
const examples = [
  {
    title: "Boolean",
    description:
      "Visualizes real-time Boolean combinations between two `tf::forms` backed by `tf::tree` hierarchies.",
    to: "boolean",
  },
  {
    title: "Isobands",
    description:
      "Shows the iso-band slices reported by running `tf::search` over a form as a plane sweeps through the mesh hierarchy.",
    to: "isobands",
  },
  {
    title: "Positioning",
    description:
      "`tf::neighbor_search` keeps pairs of metric points aligned so two forms maintain contact-quality positioning.",
    to: "positioning",
  },
  {
    title: "Scalar Field Intersections",
    description:
      "Highlights the intersection curves produced when `tf::search(form, primitive)` walks a scalar field plane through a form.",
    to: "scalar-field-intersections",
  },
  {
    title: "Collision",
    description:
      "Demonstrates pairwise `tf::search` across two forms for collision detection, exposing timing directly from the spatial hierarchy.",
    to: "collision",
  },
  {
    title: "Forms Intersections",
    description:
      "Displays intersection curves computed via `tf::search(form0, form1, ...)` and related form queries as meshes move.",
    to: "forms-intersections",
  },
];
</script>
<template>
  <div class="flex flex-row h-[calc(100vh-var(--ui-header-height,0px))]">
    <div class="w-xs flex flex-col h-full gap-2.5 overflow-y-auto p-2 border-r border-default">
      <UCard
        v-for="example in examples"
        :key="example.title"
        class="cursor-pointer flex-shrink-0"
        :ui="{
          header: 'sm:p-2 sm:px-4',
          body: 'sm:p-4 sm:px-4',
        }"
        @click="navigateTo(`/live-examples/${example.to}`)"
      >
        <template #header>
          <h3 class="text-lg font-semibold text-primary">{{ example.title }}</h3>
        </template>
        <p class="text-sm text-muted">{{ example.description }}</p>
      </UCard>
    </div>

    <div class="flex-1 h-full overflow-hidden">
      <NuxtPage class="h-full" />
    </div>
  </div>
</template>

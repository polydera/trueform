<script setup lang="ts">
const examples = [
  {
    title: "Boolean",
    description: "A live example of the boolean operation.",
    to: "boolean",
  },
  {
    title: "Isobands",
    description: "Hold shift and scroll to move the intersection curve with plane. Shows isobands visualization.",
    to: "isobands",
  },
  {
    title: "Positioning",
    description: "Drag one mesh away from the other. Their nearest neighbors will be brought back together on release.",
    to: "positioning",
  },
  {
    title: "Scalar Field Intersections",
    description: "Hold shift and scroll to move the intersection curve with plane. Shows scalar field intersections.",
    to: "scalar-field-intersections",
  },
  {
    title: "Collision",
    description: "Grab and drag a mesh to test. Intersecting meshes are highlighted in real-time.",
    to: "collision",
  },
  {
    title: "Forms Intersections",
    description: "Grab and drag a mesh to test. Intersection curves between meshes are shown.",
    to: "forms-intersections",
  },
];
</script>
<template>
  <div class="flex flex-row h-[calc(100vh-var(--ui-header-height,0px))]">
    <UDashboardSidebar class="h-full">
      <template #default>
        <div class="flex h-full flex-col gap-2.5 overflow-y-auto p-2">
          <UCard v-for="example in examples" :key="example.title" class="cursor-pointer" @click="navigateTo(`/live-examples/${example.to}`)">
            <template #header>
              <h3 class="text-lg font-bold">{{ example.title }}</h3>
            </template>
            <p class="text-sm text-muted">{{ example.description }}</p>
          </UCard>
        </div>
      </template>
    </UDashboardSidebar>
    <div class="flex-1 h-full overflow-hidden">
      <NuxtPage class="h-full" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { useMediaQuery } from "@vueuse/core";

const route = useRoute();
const isMobile = useMediaQuery("(max-width: 768px)");

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

const isSidebarOpen = ref(false);

// Close sidebar when navigating on mobile
watch(() => route.path, () => {
  if (isSidebarOpen.value) {
    isSidebarOpen.value = false;
  }
});
</script>
<template>
  <div class="relative flex flex-row h-[calc(100vh-var(--ui-header-height,0px))]">
    <!-- Floating button for mobile -->
    <UButton
      v-if="isMobile && !isSidebarOpen"
      @click="isSidebarOpen = true"
      icon="i-lucide-chevron-right"
      color="neutral"
      variant="solid"
      size="xs"
      class="fixed left-2 top-1/2 -translate-y-1/2 z-50 shadow-lg"
      :ui="{
        base: 'flex-col gap-1.5 py-2.5',
        label: '[writing-mode:vertical-rl] [text-orientation:mixed]',
      }"
      label="More examples"
    />

    <!-- Mobile Slideover -->
    <USlideover
      v-if="isMobile"
      v-model:open="isSidebarOpen"
      side="left"
      :close="false"
      :ui="{
        content: 'w-xs max-w-xs',
        body: 'p-2',
      }"
    >
      <template #body>
        <div class="flex flex-col h-full gap-2.5 overflow-y-auto">
          <div class="flex items-center justify-between mb-2">
            <h2 class="text-lg font-semibold text-primary ml-3">Live Examples</h2>
            <UButton
              icon="i-lucide-x"
              color="neutral"
              variant="ghost"
              size="sm"
              square
              @click="isSidebarOpen = false"
              aria-label="Close sidebar"
            />
          </div>

          <UCard
            v-for="example in examples"
            :key="example.title"
            class="flex-shrink-0"
            :ui="{
              root: 'cursor-pointer',
              header: 'sm:p-2 sm:px-4',
              body: 'sm:p-4 sm:px-4',
            }"
            :variant="route.path === `/live-examples/${example.to}` ? 'subtle' : 'outline'"
            @click="navigateTo(`/live-examples/${example.to}`)"
          >
            <template #header>
              <h3 class="text-lg font-semibold text-primary">{{ example.title }}</h3>
            </template>
            <p class="text-sm text-muted">{{ example.description }}</p>
          </UCard>
        </div>
      </template>
    </USlideover>

    <!-- Desktop Sidebar -->
    <div class="hidden md:flex w-xs flex-col h-full gap-2.5 overflow-y-auto p-2 border-r border-default">
      <UCard
        v-for="example in examples"
        :key="example.title"
        class="flex-shrink-0"
        :ui="{
          root: 'cursor-pointer',
          header: 'sm:p-2 sm:px-4',
          body: 'sm:p-4 sm:px-4',
        }"
        :variant="route.path === `/live-examples/${example.to}` ? 'subtle' : 'outline'"
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

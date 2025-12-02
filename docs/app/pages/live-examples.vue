<script setup lang="ts">
import { useMediaQuery } from "@vueuse/core";

const route = useRoute();
const isMobile = useMediaQuery("(max-width: 1024px)");
const isLandscape = useMediaQuery("(orientation: landscape)");

const examples = [
  {
    title: "Boolean",
    description: "Drag a mesh. The boolean updates in real time.",
    to: "boolean",
  },
  {
    title: "Slicing",
    description: "Scroll to move the plane. Cross-sections update live.",
    to: "isobands",
  },
  {
    title: "Closest Points",
    description: "Drag a mesh and release. It snaps to the nearest point instantly.",
    to: "positioning",
  },
  {
    title: "Contour Lines",
    description: "Scroll to move the plane. Contour lines update live.",
    to: "scalar-field-intersections",
  },
  {
    title: "Collision",
    description: "Drag a mesh. Contact detection runs live as you move.",
    to: "collision",
  },
  {
    title: "Mesh Intersections",
    description: "Drag a mesh. The intersection curves recompute instantly.",
    to: "forms-intersections",
  },
];

const isSidebarOpen = ref(false);

// Close sidebar when navigating on mobile
watch(
  () => route.path,
  () => {
    if (isSidebarOpen.value) {
      isSidebarOpen.value = false;
    }
  },
);
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
      :class="[
        'fixed left-2 z-50 shadow-lg',
        isLandscape ? 'bottom-2' : 'top-1/2 -translate-y-1/2',
      ]"
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
          <div class="pt-2 px-3 flex flex-col gap-2">
            <div class="flex items-center justify-between">
              <h2 class="text-2xl font-semibold text-primary">Live Examples</h2>
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
            <p class="text-sm text-muted">
              Interactive examples showcasing trueform's core features.
            </p>
            <ExampleSizePicker />
          </div>
          <USeparator class="my-1.5" />
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
    <div
      class="hidden lg:flex w-xs flex-col h-full gap-2.5 overflow-y-auto p-2 border-r border-default"
    >
      <div class="pt-2 px-3 flex flex-col gap-2">
        <h2 class="text-2xl font-semibold text-primary">Live Examples</h2>
        <p class="text-sm text-muted">Interactive examples showcasing trueform's core features.</p>
        <ExampleSizePicker />
      </div>
      <USeparator class="my-1.5" />
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

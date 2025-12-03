<script setup lang="ts">
import { useMediaQuery } from "@vueuse/core";
import type { BadgeProps } from "../types";

const props = defineProps<{
  title?: string;
  badge?: BadgeProps;
  placement?: "left" | "right";
  customPosition?: string;
}>();

const isMobile = useMediaQuery("(max-width: 1024px)");
const alignmentClass = computed(() =>
  props.placement === "right" ? "items-end text-right" : "items-start text-left",
);
const containerWidthClass = computed(() =>
  props.placement === "right"
    ? "w-fit max-w-[calc(100vw-1.5rem)] lg:max-w-sm px-3 py-1.5"
    : "max-w-[calc(100vw-1.5rem)] lg:min-w-md lg:max-w-lg p-3",
);
const positionClass = computed(() =>
  props.customPosition
    ? props.customPosition
    : `${props.placement === "right" ? "right-3 md:right-4" : "left-3 md:left-4"} absolute top-3 md:top-4`,
);
</script>

<template>
  <div
    class="z-10 rounded-lg bg-neutral-100/10 shadow-lg backdrop-blur flex flex-col gap-0.5 md:gap-2"
    :class="[alignmentClass, containerWidthClass, positionClass]"
  >
    <div v-if="title" class="flex justify-between items-start gap-3 w-full">
      <p class="font-semibold text-lg mb-2" :class="props.placement === 'right' ? 'ml-auto' : ''">{{ title }}</p>
      <UBadge
        v-if="badge"
        variant="soft"
        color="primary"
        :icon="badge.icon"
        :size="isMobile ? 'sm' : 'lg'"
        class="py-1.5 px-2 mt-0.5 lg:mt-0 inline-flex items-center gap-2 transition-[width] duration-200 ease-in-out"
      >
        <span v-if="badge.value" class="font-bold badge-text whitespace-nowrap">{{ badge.value }}</span>
        <template v-if="badge.polygons"><span v-if="badge.value">@</span><span class="badge-text whitespace-nowrap">{{ badge.polygons }}</span><UIcon name="i-lucide-triangle" /></template>
      </UBadge>
    </div>

    <div class="flex flex-col gap-2">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.badge-text {
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum";
}
</style>

<script setup lang="ts">
import { useMediaQuery } from "@vueuse/core";

interface BadgeProps {
  icon?: string;
  text: string;
}

const props = defineProps<{
  title: string;
  badge?: BadgeProps;
}>();

const isMobile = useMediaQuery("(max-width: 768px)");
</script>

<template>
  <div
    class="absolute left-3 top-3 z-10 max-w-[calc(100vw-1.5rem)] lg:min-w-md lg:max-w-lg rounded-lg p-3 bg-neutral-100/10 shadow-lg backdrop-blur flex flex-col gap-0.5 md:gap-2"
  >
    <div class="flex justify-between items-start gap-3">
      <p class="font-semibold text-lg mb-2">{{ title }}</p>
      <UBadge
        v-if="badge"
        variant="soft"
        color="primary"
        :icon="badge.icon"
        :size="isMobile ? 'sm' : 'lg'"
        class="py-1.5 px-2 mt-0.5 lg:mt-0"
      >
        <span class="font-bold">{{ badge.text }}</span>
      </UBadge>
    </div>

    <div class="flex flex-col gap-2">
      <slot />
    </div>
  </div>
</template>


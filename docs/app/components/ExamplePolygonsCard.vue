<script setup lang="ts">
import { useMediaQuery } from "@vueuse/core";

const props = defineProps<{
  meshCount: number;
  meshLabel: string;
  loading?: boolean;
}>();

const formattedLabel = computed(() => props.meshLabel);
const isMobile = useMediaQuery("(max-width: 1024px)");
const customPosition = computed(() =>
  isMobile.value ? "absolute left-1/2 -translate-x-1/2 bottom-15 top-auto" : undefined,
);
const alignment = computed(() => (isMobile.value ? "items-center text-center" : "items-end text-right"));
</script>

<template>
  <ExampleInfoCard
    v-if="!loading"
    placement="right"
    :custom-position="customPosition"
    :class="alignment"
  >
    <div class="flex flex-row items-center gap-2">
      <span class="text-[9px] lg:text-xs uppercase tracking-wide text-muted">Total polygons:</span>
      <span class="font-semibold text-xs lg:text-base">{{ formattedLabel }}</span>
    </div>
  </ExampleInfoCard>
</template>

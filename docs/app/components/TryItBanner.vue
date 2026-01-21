<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    to?: string;
    title?: string;
    description?: string;
  }>(),
  {
    to: "/live-examples/boolean",
    title: "Try it in your browser",
    description: "Interactive mesh booleans, collisions, isobands and more. No install needed.",
  },
);
</script>

<template>
  <UCard
    variant="subtle"
    :ui="{
      root: 'ring-primary/25 bg-primary/10 hover:bg-elevated/50 cursor-pointer transition-colors',
      body: 'p-4 sm:p-4 flex items-center justify-between gap-2',
    }"
    @click="navigateTo(props.to)"
  >
    <div class="flex-1">
      <h3 class="text-lg font-semibold text-primary mb-0.5">{{ props.title }}</h3>
      <p class="text-sm text-muted">{{ props.description }}</p>
    </div>
    <UIcon name="i-lucide-play" class="size-8 text-primary shrink-0" />
  </UCard>
</template>

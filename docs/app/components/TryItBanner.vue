<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    to?: string;
    title?: string;
    description?: string;
    icon?: string;
  }>(),
  {
    to: "/live-examples/boolean",
    icon: "i-lucide-play",
    title: "Try it in your browser",
    description: "Interactive mesh booleans, registration, isobands, and more. No install needed.",
  },
);
</script>

<template>
  <UCard
    class="try-it-banner"
    :ui="{
      root: 'ring-primary/25 cursor-pointer transition-colors',
      body: 'p-4 sm:p-4 flex items-center justify-between gap-2',
    }"
    @click="
      props.to.startsWith('http')
        ? navigateTo(props.to, { external: true, open: { target: '_blank' } })
        : navigateTo(props.to)
    "
  >
    <div class="flex-1">
      <h3 class="text-lg font-semibold text-primary mb-0.5">{{ props.title }}</h3>
      <p class="text-sm text-muted">{{ props.description }}</p>
    </div>
    <UIcon :name="props.icon" class="size-8 text-primary shrink-0" />
  </UCard>
</template>

<style scoped>
.try-it-banner {
  border: 1px solid var(--teal-faint);
  background: linear-gradient(135deg, var(--teal-ghost), var(--rose-ghost));
}

.try-it-banner:hover {
  border-color: var(--teal-dim);
}
</style>

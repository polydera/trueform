<script setup lang="ts">
defineProps<{
  loading: boolean;
  message?: string;
  error?: string | null;
}>();

const emit = defineEmits<{
  (event: "retry"): void;
}>();
</script>

<template>
  <Transition name="fade">
    <div
      v-if="loading || error"
      class="absolute inset-0 z-30 flex items-center justify-center bg-gradient-to-b from-background/70 via-background/60 to-background/80 backdrop-blur-md"
    >
      <UCard class="max-w-lg w-[92%] md:w-[460px] bg-background/90 border-primary/20 shadow-2xl">
        <div class="flex items-start gap-3">
          <div class="flex items-center justify-center rounded-full bg-primary/10 p-2 text-primary">
            <UIcon
              :name="error ? 'i-lucide-alert-triangle' : 'i-lucide-loader-2'"
              class="size-5"
              :class="{ 'animate-spin': loading && !error }"
            />
          </div>
          <div class="flex-1 space-y-1">
            <p class="text-sm font-semibold">
              {{ error ? "Couldn't load the live demo" : "Preparing the live demo" }}
            </p>
            <p class="text-xs text-muted">
              {{ error ?? message ?? "Fetching assets and initializing the renderer..." }}
            </p>

            <div v-if="loading && !error" class="mt-3">
              <UProgress size="lg" animation="swing" />
            </div>

            <div v-if="error" class="flex gap-2 pt-2">
              <UButton color="primary" size="sm" @click="emit('retry')">
                Retry
              </UButton>
              <UButton color="neutral" size="sm" variant="ghost" @click="emit('retry')">
                Reload assets
              </UButton>
            </div>
          </div>
        </div>
      </UCard>
    </div>
  </Transition>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>

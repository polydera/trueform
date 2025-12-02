<script setup lang="ts">
import type { ButtonProps} from "@nuxt/ui"

const { isTouchscreen } = useTouchscreen();

defineProps<{
  buttons: (ButtonProps & { keyboardShortcut?: string})[];
}>();
</script>

<template>
  <div
    class="absolute left-1/2 transform -translate-x-1/2 bottom-4 md:bottom-8 z-20 flex flex-col items-center gap-3 pointer-events-none"
    style="pointer-events: none;"
  >
    <div class="flex gap-4 pointer-events-auto">
      <UButton
        v-for="(button, index) in buttons"
        variant="subtle"
        size="lg"
        color="primary"
        v-bind="button"
      >
        <template v-if="button.keyboardShortcut && !isTouchscreen" #trailing>
          <UKbd
            variant="soft"
            color="primary"
            :value="button.keyboardShortcut"
          />
        </template>
      </UButton>
    </div>
  </div>
</template>


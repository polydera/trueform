<script setup lang="ts">
import type { ButtonProps} from "@nuxt/ui"
import { useMediaQuery } from "@vueuse/core";

const { isTouchscreen } = useTouchscreen();
const isMobile = useMediaQuery("(max-width: 1024px)");

defineProps<{
  buttons: (ButtonProps & { keyboardShortcut?: string})[];
}>();
</script>

<template>
  <div
    class="absolute left-1/2 transform -translate-x-1/2 bottom-4 lg:bottom-8 z-20 flex flex-col items-center gap-3 pointer-events-none"
    style="pointer-events: none;"
  >
    <div class="flex gap-2 lg:gap-4 pointer-events-auto">
      <UButton
        v-for="button in buttons"
        variant="subtle"
        :size="isMobile ? 'md' : 'lg'"
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


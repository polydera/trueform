<script setup lang="ts">
import type { ContentNavigationItem } from '@nuxt/content'
import { useMediaQuery } from '@vueuse/core'

const isMobile = useMediaQuery("(max-width: 1024px)");

const navigation = inject<Ref<ContentNavigationItem[]>>('navigation')
</script>

<template>
  <UContainer>
    <UPage>
      <template #left>
        <UPageAside>
          <LibPicker v-if="!isMobile" class="mb-5" />
          <UContentNavigation
            highlight
            :navigation="navigation"
          />
        </UPageAside>
      </template>

      <slot />
    </UPage>
  </UContainer>
</template>

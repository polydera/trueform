<script setup lang="ts">
import type { ContentNavigationItem } from '@nuxt/content'
import { useMediaQuery } from '@vueuse/core'

const isMobile = useMediaQuery("(max-width: 1024px)");

const navigation = inject<Ref<ContentNavigationItem[]>>('navigation')
</script>

<template>
  <UContainer class="docs-container">
    <UPage class="docs-shell">
      <template #left>
        <UPageAside class="docs-aside">
          <LibPicker v-if="!isMobile" class="mb-5" />
          <UContentNavigation
            class="docs-nav"
            highlight
            :navigation="navigation"
          />
        </UPageAside>
      </template>

      <slot />
    </UPage>
  </UContainer>
</template>

<style scoped>
.docs-container {
  position: relative;
  z-index: 1;
  padding-top: 1.5rem;
}

.docs-shell {
  gap: 1.5rem;
}

.docs-aside {
  border-right: 1px solid var(--panel-border);
  padding-right: 1.25rem;
}

.docs-nav {
  font-size: 0.9rem;
}
</style>

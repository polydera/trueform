<script setup lang="ts">
import type { ContentNavigationItem } from "@nuxt/content";
import { useMediaQuery } from "@vueuse/core";

const isMobile = useMediaQuery("(max-width: 1024px)");

const navigation = inject<Ref<ContentNavigationItem[]>>("navigation");
const route = useRoute();

const { header } = useAppConfig();

const showLibPicker = computed(() => {
  // Hide on root index page and error page
  return route.path !== "/" && route.name !== "error" && !route.path.startsWith("/live-examples");
});
</script>

<template>
  <UHeader
    class="brand-header"
    :ui="{ root: 'brand-header__root', container: 'brand-header__container', center: 'flex-1' }"
    :to="header?.to || '/'"
  >
    <UContentSearchButton
      v-if="header?.search"
      :collapsed="false"
      class="brand-search w-full"
    />

    <template v-if="header?.logo?.dark || header?.logo?.light || header?.title" #title>
      <UColorModeImage
        v-if="header?.logo?.dark || header?.logo?.light"
        :light="header?.logo?.light!"
        :dark="header?.logo?.dark!"
        :alt="header?.logo?.alt"
        class="h-6 w-auto shrink-0"
      />

      <span v-else-if="header?.title">
        {{ header.title }}
      </span>
    </template>

    <template v-else #left>
      <NuxtLink :to="header?.to || '/'" class="brand-header__logo">
        <NuxtImg src="/tf.png" class="brand-header__mark" />
        <span class="brand-header__word">trueform</span>
      </NuxtLink>
    </template>

    <template #right>
      <UContentSearchButton v-if="header?.search" class="brand-search-mobile lg:hidden" />

      <UColorModeButton v-if="header?.colorMode" />

      <GitHubStars />

      <template v-if="header?.links">
        <UButton
          v-for="(link, index) of header.links"
          :key="index"
          v-bind="{ color: 'neutral', variant: 'ghost', ...link }"
        />
      </template>
    </template>

    <template #body>
      <LibPicker v-if="showLibPicker && isMobile" class="mb-4" />
      <UContentNavigation class="brand-mobile-nav" highlight :navigation="navigation" />
    </template>
  </UHeader>
</template>

<style scoped>
.brand-header {
  position: sticky;
  top: 0;
  z-index: 15;
  border-bottom: 1px solid var(--panel-border);
  background: rgba(12, 21, 19, 0.78);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}

.brand-header__logo {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  color: var(--text);
  text-decoration: none;
}

.brand-header__mark {
  width: auto;
  height: 1.45rem;
  filter: drop-shadow(0 0 14px var(--teal-faint));
}

.brand-header__word {
  font-size: 1.35rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.brand-mobile-nav {
  border: 1px solid var(--panel-border);
  border-radius: var(--ui-radius);
  background: var(--panel-bg);
  padding: 0.75rem;
}
</style>

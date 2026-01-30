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
  <UHeader :ui="{ center: 'flex-1' }" :to="header?.to || '/'">
    <UContentSearchButton v-if="header?.search" :collapsed="false" class="w-full" />

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
      <NuxtLink :to="header?.to || '/'" class="flex items-center gap-1">
        <NuxtImg src="/tf.png" class="w-auto h-6 shrink-0" />
        <h1 class="text-2xl font-bold">trueform</h1>
      </NuxtLink>
    </template>

    <template #right>
      <UContentSearchButton v-if="header?.search" class="lg:hidden" />

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
      <UContentNavigation highlight :navigation="navigation" />
    </template>
  </UHeader>
</template>

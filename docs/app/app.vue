<script setup lang="ts">
const { seo } = useAppConfig();
const config = useRuntimeConfig();

const { navigation, files } = useLibraryCollection();

const route = useRoute();
const colorMode = useColorMode();
colorMode.preference = "dark";

// Get site URL from site config (same as nuxt.config.ts)
const site = useSiteConfig();

// Build canonical URL
const canonicalUrl = computed(() => {
  const path = route.path === "/" ? "" : route.path;
  return `${site.url}${path}`;
});

useHead({
  meta: [
    { name: "viewport", content: "width=device-width, initial-scale=1" },
    { key: "theme-color", name: "theme-color", content: "#0c1513" },
  ],
  link: [
    { rel: "icon", type: "image/png", href: `${config.app.baseURL}tf.png` },
    { rel: "canonical", href: canonicalUrl },
  ],
  htmlAttrs: {
    lang: "en",
  },
});

useSeoMeta({
  titleTemplate: `%s - ${seo?.siteName}`,
  ogSiteName: seo?.siteName,
  ogUrl: canonicalUrl,
  twitterCard: "summary_large_image",
});

// Add Schema.org structured data
// Organization is auto-generated from site.identity in nuxt.config.ts
useSchemaOrg([
  defineWebSite({
    name: "trueform",
    description:
      "Geometry library for real-time spatial queries, mesh booleans, and topology. C++ header-only with Python bindings.",
  }),
  defineWebPage(),
]);

provide("navigation", navigation);
</script>

<template>
  <UApp>
    <div class="site-shell">
      <div class="site-grid" />
      <span class="corner-mark corner-mark--tl" />
      <span class="corner-mark corner-mark--tr" />
      <span class="corner-mark corner-mark--bl" />
      <span class="corner-mark corner-mark--br" />

      <div class="site-layer">
        <NuxtLoadingIndicator />

        <AppHeader />

        <UMain>
          <NuxtLayout>
            <NuxtPage />
          </NuxtLayout>
        </UMain>

        <AppFooter v-if="!route.path.startsWith('/live-examples')" />

        <ClientOnly>
          <LazyUContentSearch :files="files" :navigation="navigation" />
        </ClientOnly>
      </div>
    </div>
  </UApp>
</template>

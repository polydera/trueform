<script setup lang="ts">
const { seo } = useAppConfig();
const config = useRuntimeConfig();

const { data: navigation } = await useAsyncData("navigation", () =>
  queryCollectionNavigation("docs"),
);
const { data: files } = useLazyAsyncData("search", () => queryCollectionSearchSections("docs"), {
  server: false,
});

useHead({
  meta: [{ name: "viewport", content: "width=device-width, initial-scale=1" }],
  link: [{ rel: "icon", type: "image/png", href: `${config.app.baseURL}tf.png` }],
  htmlAttrs: {
    lang: "en",
  },
});

useSeoMeta({
  titleTemplate: `%s - ${seo?.siteName}`,
  ogSiteName: seo?.siteName,
  // twitterCard: "summary_large_image",
});

provide("navigation", navigation);
</script>

<template>
  <UApp>
    <NuxtLoadingIndicator />

    <AppHeader />

    <UMain>
      <NuxtLayout>
        <NuxtPage />
      </NuxtLayout>
    </UMain>

    <AppFooter />

    <ClientOnly>
      <LazyUContentSearch :files="files" :navigation="navigation" />
    </ClientOnly>
  </UApp>
</template>

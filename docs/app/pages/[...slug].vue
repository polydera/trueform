<script setup lang="ts">
import type { ContentNavigationItem } from "@nuxt/content";
import { findPageHeadline } from "@nuxt/content/utils";

definePageMeta({
  layout: "docs",
});

const route = useRoute();
const { collection, library } = useLibraryCollection();
const { toc } = useAppConfig();
const navigation = inject<Ref<ContentNavigationItem[]>>("navigation");

const { data: page } = await useAsyncData(
  () => `${collection.value}-${route.path}`,
  () => queryCollection(collection.value).path(route.path).first(),
);
if (!page.value) {
  throw createError({ statusCode: 404, statusMessage: "Page not found", fatal: true });
}

const { data: surround } = await useAsyncData(
  () => `${collection.value}-${route.path}-surround`,
  () => {
    return queryCollectionItemSurroundings(collection.value, route.path, {
      fields: ["description"],
    });
  },
);

const title = page.value.seo?.title || page.value.title;
const description = page.value.seo?.description || page.value.description;

const headline = computed(() => `${findPageHeadline(navigation?.value, page.value?.path)} | ${library.value === "cpp" ? "C++" : "PY"}`);

// Generate OG image using nuxt-og-image with headline
defineOgImageComponent("Docs", {
  title,
  description,
  headline: headline.value,
});

useSeoMeta({
  title,
  ogTitle: title,
  description,
  ogDescription: description,
});

// Build breadcrumb list for Schema.org
const breadcrumbs = computed(() => {
  const items: { name: string; item: string }[] = [];
  const pathParts = route.path.split("/").filter(Boolean);

  let currentPath = "";
  pathParts.forEach((part, index) => {
    currentPath += `/${part}`;
    const navItem = navigation?.value
      ?.flatMap((n) => [n, ...(n.children || [])])
      .find((item) => item.path === currentPath);

    items.push({
      name: navItem?.title || part,
      item: `https://trueform.polydera.com${currentPath}`,
    });
  });

  return items;
});

// Add breadcrumb structured data
useSchemaOrg([
  defineBreadcrumb({
    itemListElement: breadcrumbs.value,
  }),
]);
</script>

<template>
  <UPage v-if="page">
    <UPageHeader :title="page.title" :description="page.description" :headline="headline">
      <template #links>
        <UButton v-for="(link, index) in page.links" :key="index" v-bind="link" />

        <PageHeaderLinks />
      </template>
    </UPageHeader>

    <UPageBody>
      <ContentRenderer v-if="page" :value="page" />

      <USeparator v-if="surround?.length" />

      <UContentSurround :surround="surround" />
    </UPageBody>

    <template v-if="page?.body?.toc?.links?.length" #right>
      <UContentToc :title="toc?.title" :links="page.body?.toc?.links"> </UContentToc>
    </template>
  </UPage>
</template>

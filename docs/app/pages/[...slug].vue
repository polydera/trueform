<script setup lang="ts">
import type { ContentNavigationItem } from "@nuxt/content";
import { findPageHeadline } from "@nuxt/content/utils";

definePageMeta({
  layout: "docs",
});

const route = useRoute();
const { collection } = useLibraryCollection();
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

useSeoMeta({
  title,
  ogTitle: title,
  description,
  ogDescription: description,
});

const headline = computed(() => findPageHeadline(navigation?.value, page.value?.path));

// defineComponent("Docs", {
//   headline: headline.value,
// });
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

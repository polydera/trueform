import type { ContentNavigationItem } from "@nuxt/content";

export const useTransformedNavigation = (
  navigation: Ref<ContentNavigationItem[] | null | undefined>,
) => {
  return computed(() => {
    if (!navigation.value || navigation.value.length === 0) {
      return navigation.value ?? null;
    }

    // Remove the first level (cpp/py parent) by extracting its children
    const firstItem = navigation.value[0];
    if (firstItem?.children) {
      return firstItem.children;
    }

    return navigation.value;
  });
};

<script setup lang="ts">
import type { ContentNavigationItem } from "@nuxt/content";

const { library } = useLibraryCollection();
const route = useRoute();

type Library = "cpp" | "py" | "ts";

const collectionByLibrary = {
  cpp: "docsCpp",
  py: "docsPy",
  ts: "docsTs",
} as const;

const items = [
  {
    label: "C++",
    shortLabel: "C++",
    value: "cpp" as const,
  },
  {
    label: "Python",
    shortLabel: "Python",
    value: "py" as const,
  },
  {
    label: "TypeScript",
    shortLabel: "TS",
    value: "ts" as const,
  },
];

const { data: libraryNavigation } = await useAsyncData("lib-picker-navigation", async () => {
  const [cpp, py, ts] = await Promise.all([
    queryCollectionNavigation(collectionByLibrary.cpp),
    queryCollectionNavigation(collectionByLibrary.py),
    queryCollectionNavigation(collectionByLibrary.ts),
  ]);

  return { cpp, py, ts };
});

const flattenPaths = (items: ContentNavigationItem[] | null | undefined): string[] => {
  if (!items) {
    return [];
  }

  return items.flatMap((item) => [
    item.path,
    ...flattenPaths(item.children),
  ]).filter((path): path is string => Boolean(path));
};

const libraryPaths = computed(() => ({
  cpp: new Set(flattenPaths(libraryNavigation.value?.cpp)),
  py: new Set(flattenPaths(libraryNavigation.value?.py)),
  ts: new Set(flattenPaths(libraryNavigation.value?.ts)),
}));

const getTargetPath = (value: Library) => {
  const newLibrary = value;

  // If we're on a library-specific path, try to find the equivalent page
  const currentPath = route.path;
  const pathParts = currentPath.split("/");

  if (pathParts[1] === "cpp" || pathParts[1] === "py" || pathParts[1] === "ts") {
    const newPath = `/${newLibrary}${currentPath.slice(pathParts[1].length + 1)}`;

    // Check if the page exists in the new collection
    if (libraryPaths.value[newLibrary].has(newPath)) {
      return newPath;
    }
  }

  // Fallback to getting-started
  return `/${newLibrary}/getting-started`;
};
</script>
<template>
  <div class="brand-lib-picker" role="tablist" aria-label="Documentation language">
    <NuxtLink
      v-for="item in items"
      :key="item.value"
      :to="getTargetPath(item.value)"
      class="brand-lib-picker__button"
      :class="{ 'brand-lib-picker__button--active': library === item.value }"
      role="tab"
      :aria-selected="library === item.value"
      :aria-label="item.label"
    >
      <span>{{ item.shortLabel }}</span>
    </NuxtLink>
  </div>
</template>

<style scoped>
.brand-lib-picker {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  width: 100%;
  border-bottom: 1px solid var(--panel-border);
  padding-bottom: 0.75rem;
}

.brand-lib-picker__button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
  min-height: 1.75rem;
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  text-decoration: none;
  transition: color 160ms ease, opacity 160ms ease;
}

.brand-lib-picker__button + .brand-lib-picker__button::before {
  margin-right: 0.65rem;
  color: var(--rose-dim);
  content: "/";
  opacity: 0.7;
}

.brand-lib-picker__button span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.brand-lib-picker__button--active {
  color: var(--teal);
}

.brand-lib-picker__button:hover {
  color: var(--text);
}

@media (max-width: 420px) {
  .brand-lib-picker__button span {
    display: none;
  }
}
</style>

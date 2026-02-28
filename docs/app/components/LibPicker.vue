<script setup lang="ts">
const { library, collection } = useLibraryCollection();
const route = useRoute();

const items = [
  {
    icon: "i-vscode-icons:file-type-cpp",
    value: "cpp",
  },
  {
    icon: "i-vscode-icons:file-type-python",
    value: "py",
  },
  {
    icon: "i-vscode-icons:file-type-typescript",
    value: "ts",
  },
];

const router = useRouter();
const handleChange = async (value: string | number) => {
  const newLibrary = value as "cpp" | "py" | "ts";
  const newCollection = newLibrary === "cpp" ? "docsCpp" : newLibrary === "py" ? "docsPy" : "docsTs";

  // If we're on a library-specific path, try to find the equivalent page
  const currentPath = route.path;
  const pathParts = currentPath.split("/");

  if (pathParts[1] === "cpp" || pathParts[1] === "py" || pathParts[1] === "ts") {
    // Replace the library prefix
    const newPath = `/${newLibrary}${currentPath.slice(pathParts[1].length + 1)}`;

    // Check if the page exists in the new collection
    try {
      const page = await queryCollection(newCollection).path(newPath).first();
      if (page) {
        router.push(newPath);
        return;
      }
    } catch {
      // Page doesn't exist, fall through to getting-started
    }
  }

  // Fallback to getting-started
  router.push(`/${newLibrary}/getting-started`);
};
</script>
<template>
  <UTabs
    :items="items"
    :model-value="library"
    :content="false"
    size="sm"
    color="primary"
    @update:model-value="handleChange"
  />
</template>

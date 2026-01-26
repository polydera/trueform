export const useLibraryCollection = () => {
  const route = useRoute();

  // Determine library from route path, fallback to storage
  const library = computed(() => {
    const pathLibrary = route.path.split("/")[1];
    if (pathLibrary === "cpp" || pathLibrary === "py") {
      return pathLibrary as "cpp" | "py";
    }
    return "cpp";
  });

  const collection = computed(() => (library.value === "cpp" ? "docsCpp" : "docsPy"));

  // Fetch navigation data
  const { data: navigation } = useAsyncData(
    () => `navigation-${collection.value}`,
    () => queryCollectionNavigation(collection.value),
  );
  const transformedNavigation = useTransformedNavigation(navigation);

  // Fetch search files data
  const { data: files } = useLazyAsyncData(
    () => `search-${collection.value}`,
    () => queryCollectionSearchSections(collection.value),
    {
      server: false,
    },
  );

  return { library, collection, navigation: transformedNavigation, files };
};

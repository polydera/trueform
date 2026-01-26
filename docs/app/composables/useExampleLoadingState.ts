export function useExampleLoadingState(defaultMessage = "Loading compute engine…") {
  const isLoading = ref(false);
  const loadingMessage = ref(defaultMessage);
  const loadingError = ref<string | null>(null);

  const resetLoading = (message = defaultMessage) => {
    isLoading.value = true;
    loadingMessage.value = message;
    loadingError.value = null;
  };

  const setLoadingMessage = (message: string) => {
    loadingMessage.value = message;
    loadingError.value = null;
  };

  const failLoading = (error: unknown) => {
    loadingError.value = error instanceof Error ? error.message : String(error);
    isLoading.value = false;
  };

  const finishLoading = () => {
    isLoading.value = false;
  };

  return {
    isLoading,
    loadingMessage,
    loadingError,
    resetLoading,
    setLoadingMessage,
    failLoading,
    finishLoading,
  };
}

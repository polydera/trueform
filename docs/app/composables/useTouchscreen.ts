export const useTouchscreen = () => {
  const isTouchscreen = computed(() => {
    if (import.meta.server) {
      // SSR-safe fallback
      return false
    }

    // Client-only browser checks
    return (
      'ontouchstart' in window ||
      navigator.maxTouchPoints > 0 ||
      (navigator as any).msMaxTouchPoints > 0
    )
  });
  return { isTouchscreen };
};

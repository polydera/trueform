<script setup lang="ts">
import type { ButtonProps } from "@nuxt/ui";
import type { BadgeProps } from "@/types";

const props = withDefaults(
  defineProps<{
    title?: string;
    badge?: BadgeProps;
    polygonLabel?: string;
    loading: boolean;
    loadingMessage?: string;
    loadingError?: string | null;
    actionButtons?: (ButtonProps & { keyboardShortcut?: string })[];
    infoPlacement?: "left" | "right";
    infoPosition?: string;
    hideInfoDuringLoading?: boolean;
  }>(),
  {
    infoPlacement: "left",
    hideInfoDuringLoading: false,
  },
);

const emit = defineEmits<{
  (event: "retry"): void;
}>();

const resolvedBadge = computed(() => {
  const baseBadge =
    props.badge ??
    (props.polygonLabel
      ? {
          icon: "i-lucide-triangle",
          value: props.polygonLabel,
        }
      : undefined);

  if (!baseBadge) {
    return undefined;
  }

  if (props.polygonLabel && !baseBadge.polygons) {
    return { ...baseBadge, polygons: props.polygonLabel };
  }

  return baseBadge;
});
const hasActions = computed(() => (props.actionButtons?.length ?? 0) > 0);
const showInfoCard = computed(() => !props.hideInfoDuringLoading || !props.loading);
</script>

<template>
  <div class="flex flex-col w-full h-full">
    <div class="flex flex-row flex-1 relative min-h-0">
      <ExampleInfoCard
        v-if="showInfoCard"
        :title="title"
        :badge="resolvedBadge"
        :placement="infoPlacement"
        :custom-position="infoPosition"
      >
        <slot name="info" />
      </ExampleInfoCard>

      <div class="flex flex-col md:flex-row w-full">
        <slot name="containers">
          <div class="h-full flex-1 min-h-0 w-screen md:w-full" />
        </slot>
      </div>

      <ExampleLoadingOverlay
        :loading="loading"
        :message="loadingMessage"
        :error="loadingError"
        @retry="emit('retry')"
      />
      <ExampleActionButtons v-if="hasActions" :buttons="actionButtons!" />
    </div>
  </div>
</template>

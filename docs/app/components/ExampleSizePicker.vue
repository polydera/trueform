<script setup lang="ts">
import { useMeshSelection } from "@/composables/useMeshSelection";
import type { SelectItem } from "@nuxt/ui";

const { meshSize, meshSizeOptions } = useMeshSelection();

// Cast to mutable array for USelect component
const selectItems = computed(() => meshSizeOptions as unknown as SelectItem[]);

type MeshSizeItem = SelectItem & { size: string };
</script>
<template>
  <div class="flex flex-col gap-1">
    <h3 class="text-md font-semibold text-muted">Choose mesh size:</h3>
    <USelect v-model="meshSize" :items="selectItems">
      <template #trailing>
        <UBadge
          :label="(meshSizeOptions.find(size => size.value === meshSize) as MeshSizeItem | undefined)?.size"
          variant="soft"
          size="sm"
          color="neutral"
        />
        <UIcon name="i-lucide-chevron-down" class="size-4 ml-1.5" />
      </template>
      <template #item-trailing="{ item }">
        <UBadge
          v-if="item"
          :label="(item as MeshSizeItem).size"
          variant="soft"
          size="sm"
          color="neutral"
        />
      </template>
    </USelect>
  </div>
</template>

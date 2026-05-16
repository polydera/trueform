<script setup lang="ts">
import { useClipboard } from "@vueuse/core";

const route = useRoute();
const toast = useToast();
const { copy, copied } = useClipboard();
const site = useSiteConfig();
const isCopying = ref(false);

const mdPath = computed(() => `${site.url}/raw${route.path}.md`);

const items = [
  {
    label: "Copy Markdown link",
    icon: "i-lucide-link",
    onSelect() {
      copy(mdPath.value);
      toast.add({
        title: "Copied to clipboard",
        icon: "i-lucide-check-circle",
      });
    },
  },
  {
    label: "View as Markdown",
    icon: "i-simple-icons:markdown",
    target: "_blank",
    to: `/raw${route.path}.md`,
  },
  {
    label: "Open in ChatGPT",
    icon: "i-simple-icons:openai",
    target: "_blank",
    to: `https://chatgpt.com/?hints=search&q=${encodeURIComponent(`Read ${mdPath.value} so I can ask questions about it.`)}`,
  },
  {
    label: "Open in Claude",
    icon: "i-simple-icons:anthropic",
    target: "_blank",
    to: `https://claude.ai/new?q=${encodeURIComponent(`Read ${mdPath.value} so I can ask questions about it.`)}`,
  },
];

async function copyPage() {
  isCopying.value = true;
  copy(await $fetch<string>(`/raw${route.path}.md`));
  isCopying.value = false;
}
</script>

<template>
  <div class="brand-page-links">
    <UDropdownMenu
      :items="items"
      :content="{
        align: 'end',
        side: 'bottom',
        sideOffset: 8,
      }"
      :ui="{
        content: 'w-48',
      }"
    >
      <UButton label="Copy page" color="neutral" variant="outline" class="brand-page-links__button">
        <template #trailing>
          <div class="flex items-center gap-0.5">
            <UIcon name="i-lucide-copy" />
            <UIcon name="i-simple-icons:openai" />
            <UIcon name="i-simple-icons:anthropic" />
          </div>
        </template>
      </UButton>
    </UDropdownMenu>
  </div>
</template>

<style scoped>
.brand-page-links {
  display: inline-flex;
  border-radius: var(--ui-radius);
  box-shadow: 0 0 0 1px var(--rose-faint);
}

.brand-page-links__button {
  background: var(--rose-ghost);
}
</style>

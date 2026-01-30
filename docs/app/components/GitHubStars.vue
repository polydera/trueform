<script setup lang="ts">
const stars = ref<number | null>(null)

onMounted(async () => {
  try {
    const res = await fetch('https://api.github.com/repos/polydera/trueform')
    const data = await res.json()
    stars.value = data.stargazers_count
  } catch {
    stars.value = null
  }
})
</script>

<template>
  <UButton
    to="https://github.com/polydera/trueform"
    target="_blank"
    color="neutral"
    variant="ghost"
    aria-label="Star on GitHub"
  >
    <UIcon name="i-lucide-star" class="size-5" />
    <span v-if="stars !== null">{{ stars }}</span>
  </UButton>
</template>

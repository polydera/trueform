<template>
  <section class="brand-section">
    <div class="brand-inner">
      <p class="brand-kicker">Integrations</p>
      <h2 class="brand-title brand-integration-grid__title">
        One <span class="brand-title__primary">engine</span>.
        Four <span class="brand-title__secondary">integrations</span>.
      </h2>

      <div class="brand-integration-grid__grid">
        <NuxtLink
          v-for="item in integrations"
          :key="item.title"
          :to="item.to"
          class="glass-panel glass-panel--interactive brand-integration-card"
        >
          <UIcon :name="item.icon" class="brand-integration-card__icon" />
          <div>
            <h3>{{ item.title }}</h3>
            <p>{{ item.description }}</p>
          </div>
          <UIcon name="i-lucide-arrow-right" class="brand-integration-card__arrow" />
        </NuxtLink>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
const integrations = [
  {
    title: "Python",
    description: "NumPy for geometry. Fast and exact mesh booleans, spatial queries, and topology.",
    icon: "i-vscode-icons:file-type-python",
    to: "/py/getting-started",
  },
  {
    title: "TypeScript",
    description: "Native geometry processing for web apps, Node.js, and browser-native workflows.",
    icon: "i-vscode-icons:file-type-typescript-official",
    to: "/ts/getting-started",
  },
  {
    title: "VTK",
    description: "trueform performance inside VTK pipelines with filters and utility functions.",
    icon: "i-lucide-layers",
    to: "/cpp/vtk",
  },
  {
    title: "Blender",
    description: "Cached meshes and live-preview geometry processing for Blender add-ons.",
    icon: "i-vscode-icons:file-type-blender",
    to: "/py/blender",
  },
];
</script>

<style scoped>
.brand-integration-grid__title {
  margin-top: 0.75rem;
}

.brand-integration-grid__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1.25rem;
  margin-top: 2rem;
}

.brand-integration-card {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: start;
  padding: 1.4rem;
}

.brand-integration-card__icon {
  width: 1.45rem;
  height: 1.45rem;
  color: var(--teal);
}

.brand-integration-card h3 {
  color: var(--text);
  font-weight: 750;
}

.brand-integration-card p {
  margin-top: 0.35rem;
  color: var(--text-dim);
  font-size: 0.88rem;
  line-height: 1.6;
}

.brand-integration-card__arrow {
  width: 1rem;
  height: 1rem;
  color: var(--rose);
}

@media (max-width: 767px) {
  .brand-integration-grid__grid {
    grid-template-columns: 1fr;
  }
}
</style>

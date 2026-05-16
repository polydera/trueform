<template>
  <section class="brand-section">
    <div class="brand-inner">
      <p class="brand-kicker">Why trueform</p>

      <div class="brand-feature-grid__grid">
        <div
          v-for="feature in features"
          :key="feature.title"
          class="brand-feature-card"
        >
          <div class="brand-feature-card__head">
            <UIcon :name="feature.icon" class="brand-feature-card__icon" />
            <h3>{{ feature.title }}</h3>
          </div>
          <p>{{ feature.description }}</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
const features = [
  {
    title: "Easy to use",
    description: "Zero-copy views in C++, NumPy arrays in Python, vectorized NDArrays in the browser.",
    icon: "i-lucide-sparkles",
  },
  {
    title: "Robust by design",
    description: "Exact predicates and canonical topology for non-manifold, messy, real-world meshes.",
    icon: "i-lucide-shield-check",
  },
  {
    title: "Real-time performance",
    description: "Interactive speed on million-polygon meshes with parallel algorithms underneath.",
    icon: "i-lucide-zap",
  },
  {
    title: "Queries & topology",
    description: "Spatial trees, collision, distance, ray casting, k-NN, boundaries, and components.",
    icon: "i-lucide-tree-pine",
  },
  {
    title: "Arrangements & booleans",
    description: "Multi-mesh arrangements, self-intersection resolution, booleans, and curves.",
    icon: "i-lucide-scissors",
  },
  {
    title: "Remeshing & registration",
    description: "Decimation, isotropic remeshing, point-cloud alignment, OBB fitting, and ICP.",
    icon: "i-lucide-triangle",
  },
];
</script>

<style scoped>
.brand-feature-grid__grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 2.5rem 2rem;
  margin-top: 2rem;
}

.brand-feature-card {
  display: grid;
  grid-template-columns: 1.25rem 1fr;
  gap: 0 0.6rem;
  color: inherit;
}

.brand-feature-card__head {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.brand-feature-card__icon {
  flex-shrink: 0;
  width: 1.25rem;
  height: 1.25rem;
  color: var(--teal);
  opacity: 0.7;
}

.brand-feature-card h3 {
  color: var(--text);
  font-size: 1rem;
  font-weight: 600;
}

.brand-feature-card p {
  grid-column: 2;
  margin-top: 0.4rem;
  color: var(--text-dim);
  font-size: 0.85rem;
  line-height: 1.6;
}

@media (max-width: 1023px) {
  .brand-feature-grid__grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 639px) {
  .brand-feature-grid__grid {
    grid-template-columns: 1fr;
  }
}
</style>

<template>
  <section class="brand-hero">
    <div class="brand-inner brand-hero__inner">
      <div class="brand-hero__copy">
        <p class="brand-kicker">C++ · Python · TypeScript</p>
        <h1 class="brand-hero__title">
          The <span>STL</span> for <strong>Geometry</strong>
        </h1>
        <p class="brand-hero__description">
          Fast and exact mesh booleans, spatial queries, arrangements, registration, and remeshing.
          Composable algorithms on your data. One engine across C++, Python, and TypeScript.
        </p>

        <div class="brand-hero__actions">
          <UButton to="/cpp/getting-started" size="lg" variant="subtle" trailing-icon="i-vscode-icons:file-type-cpp">
            C++
          </UButton>
          <UButton to="/py/getting-started" size="lg" variant="subtle" trailing-icon="i-vscode-icons:file-type-python">
            Python
          </UButton>
          <UButton to="/ts/getting-started" size="lg" variant="subtle" trailing-icon="i-vscode-icons:file-type-typescript-official">
            TypeScript
          </UButton>
          <UButton to="/live-examples/boolean" size="lg" color="primary" variant="solid" icon="i-lucide-play">
            Try it live
          </UButton>
        </div>
      </div>

      <div class="brand-hero__visual">
        <ChartCarousel />
      </div>
    </div>
    <div class="brand-hero__fade" />
  </section>
</template>

<style scoped>
.brand-hero {
  position: relative;
  min-height: calc(100svh - 4rem);
  display: flex;
  align-items: center;
  overflow: hidden;
  padding-block: 5rem 4rem;
}

.brand-hero__inner {
  display: grid;
  grid-template-columns: minmax(0, 0.9fr) minmax(22rem, 1.1fr);
  gap: 3rem;
  align-items: center;
}

.brand-hero__title {
  margin-top: 1rem;
  color: var(--text);
  font-size: clamp(3rem, 7vw, 6rem);
  font-weight: 850;
  letter-spacing: -0.04em;
  line-height: 0.95;
}

.brand-hero__title span {
  color: var(--teal);
  text-shadow: 0 0 44px var(--teal-faint);
}

.brand-hero__title strong {
  display: block;
  color: var(--rose);
  font-weight: inherit;
  text-shadow: 0 0 44px var(--rose-faint);
}

.brand-hero__description {
  max-width: 42rem;
  margin-top: 1.5rem;
  color: var(--text-dim);
  font-size: clamp(1rem, 1.6vw, 1.2rem);
  line-height: 1.7;
}

.brand-hero__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.brand-hero__actions {
  margin-top: 2rem;
}

.brand-hero__visual {
  min-width: 0;
}

.brand-hero__fade {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  height: 10rem;
  background: linear-gradient(to bottom, transparent, var(--bg-0));
  pointer-events: none;
}

@media (max-width: 1023px) {
  .brand-hero {
    min-height: auto;
    padding-top: 4rem;
  }

  .brand-hero__inner {
    grid-template-columns: 1fr;
  }
}
</style>

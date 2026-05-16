<template>
  <section class="brand-section brand-cta-section">
    <div class="brand-inner">
      <div class="glass-panel brand-cta">
        <p class="brand-kicker">Start now</p>
        <h2 class="brand-title">
          From install to <span class="brand-title__primary">mesh booleans</span> in minutes.
        </h2>
        <div class="brand-cta__actions">
          <UButton to="/cpp/getting-started" size="lg" variant="subtle" icon="i-vscode-icons:file-type-cpp" trailing-icon="i-lucide-arrow-right">
            C++
          </UButton>
          <UButton to="/py/getting-started" size="lg" variant="subtle" icon="i-vscode-icons:file-type-python" trailing-icon="i-lucide-arrow-right">
            Python
          </UButton>
          <UButton to="/ts/getting-started" size="lg" variant="subtle" icon="i-vscode-icons:file-type-typescript-official" trailing-icon="i-lucide-arrow-right">
            TypeScript
          </UButton>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.brand-cta-section {
  padding-bottom: 2rem;
}

.brand-cta {
  padding: clamp(2rem, 5vw, 4rem);
  text-align: center;
}

.brand-cta .brand-title {
  max-width: 48rem;
  margin: 0.75rem auto 0;
}

.brand-cta__actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.75rem;
  margin-top: 2rem;
}
</style>

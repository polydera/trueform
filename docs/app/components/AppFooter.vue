<script setup lang="ts">
const { footer } = useAppConfig()
</script>

<template>
  <footer class="brand-footer">
    <div class="brand-inner">
      <div class="brand-footer__grid">
        <div class="brand-footer__brand">
          <NuxtLink to="/" class="brand-footer__logo">
            <NuxtImg src="/tf.png" class="brand-footer__mark" />
            <span>trueform</span>
          </NuxtLink>
          <p>The STL for Geometry.</p>
          <p class="brand-footer__polydera">By Polydera.</p>
        </div>

        <div class="brand-footer__col">
          <p class="brand-footer__label">Docs</p>
          <NuxtLink to="/cpp/getting-started">C++</NuxtLink>
          <NuxtLink to="/py/getting-started">Python</NuxtLink>
          <NuxtLink to="/ts/getting-started">TypeScript</NuxtLink>
        </div>

        <div class="brand-footer__col">
          <p class="brand-footer__label">Explore</p>
          <NuxtLink to="/cpp/benchmarks">Benchmarks</NuxtLink>
          <NuxtLink to="/live-examples/boolean">Live examples</NuxtLink>
          <a href="https://github.com/polydera/trueform" target="_blank">GitHub</a>
        </div>

        <div class="brand-footer__col">
          <p class="brand-footer__label">Company</p>
          <a href="https://polydera.com" target="_blank">Polydera</a>
          <span>{{ footer.credits }}</span>
        </div>
      </div>
    </div>
  </footer>
</template>

<style scoped>
.brand-footer {
  position: relative;
  z-index: 1;
  margin-top: 4rem;
  border-top: 1px solid var(--panel-border);
  padding-block: 3rem 2rem;
  background: rgba(12, 21, 19, 0.38);
}

.brand-footer__grid {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr;
  gap: 2rem;
}

.brand-footer__logo {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text);
  font-weight: 800;
  text-decoration: none;
}

.brand-footer__mark {
  width: auto;
  height: 1.1rem;
}

.brand-footer__brand p,
.brand-footer__col a,
.brand-footer__col span {
  color: var(--text-muted);
  font-size: 0.82rem;
  line-height: 1.6;
  text-decoration: none;
}

.brand-footer__brand p {
  margin-top: 0.65rem;
}

.brand-footer__polydera {
  color: var(--rose-dim) !important;
}

.brand-footer__col {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.brand-footer__label {
  margin-bottom: 0.25rem;
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.brand-footer__col a:hover {
  color: var(--teal);
}

@media (max-width: 767px) {
  .brand-footer__grid {
    grid-template-columns: 1fr 1fr;
  }

  .brand-footer__brand {
    grid-column: 1 / -1;
  }
}
</style>

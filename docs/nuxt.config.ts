// https://nuxt.com/docs/api/configuration/nuxt-config
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const trueformPkg = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("./node_modules/@polydera/trueform/package.json", import.meta.url)),
    "utf-8",
  ),
);

export default defineNuxtConfig({
  runtimeConfig: {
    public: {
      trueformVersion: trueformPkg.version as string,
    },
  },
  modules: [
    "@nuxt/eslint",
    "@nuxt/image",
    "@nuxt/ui",
    "nuxt-llms",
    "@nuxtjs/robots",
    "@nuxtjs/sitemap",
    "@nuxt/content",
    "nuxt-og-image",
    "nuxt-schema-org",
    "./modules/copy-files",
    "@nuxt/fonts",
  ],

  devtools: {
    enabled: false,
  },

  colorMode: {
    preference: "dark",
    fallback: "dark",
  },

  css: ["~/assets/css/main.css"],

  content: {
    build: {
      markdown: {
        toc: {
          searchDepth: 1,
        },
        highlight: {
          langs: ["cpp", "python", "typescript", "bash", "cmake"],
        },
      },
    },
  },

  compatibilityDate: "2024-09-23",

  // OG Image configuration
  site: {
    url: process.env.NUXT_PUBLIC_SITE_URL || "https://trueform.polydera.com",
    name: "trueform",
    identity: {
      type: "Organization",
      name: "XLAB Medical",
      url: "https://trueform.polydera.com",
      logo: "https://trueform.polydera.com/tf.png",
    },
  },

  ogImage: {
    security: {
      secret: "af94914efaa6630c19a796c440b8d39d248e81d92f5ea5ce1b8c639bd0153e3e",
    },
  },

  // Sitemap configuration
  sitemap: {
    autoLastmod: true,
    zeroRuntime: true,
  },

  nitro: {
    esbuild: {
      options: {
        target: "es2022",
      },
    },
    prerender: {
      routes: ["/"],
      crawlLinks: true,
      autoSubfolderIndex: false,
    },
    routeRules: {
      "/**": {
        headers: {
          "Cross-Origin-Embedder-Policy": "require-corp",
          "Cross-Origin-Opener-Policy": "same-origin",
          "Cross-Origin-Resource-Policy": "cross-origin",
        },
      },
    },
    preset: "cloudflare_pages",
    cloudflare: {
      deployConfig: true,
      nodeCompat: true,
      pages: {
        routes: {
          exclude: ["/cpp/*", "/py/*", "/ts/*", "/live-examples/*"],
        },
      },
    },
  },

  eslint: {
    config: {
      stylistic: {
        commaDangle: "never",
        braceStyle: "1tbs",
      },
    },
  },

  icon: {
    provider: "iconify",
    collections: ["lucide", "simple-icons", "vscode-icons", "material-icon-theme"],
    fetchTimeout: 5000,
  },

  vite: {
    server: {
      headers: {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "cross-origin",
      },
    },
    worker: {
      format: "es",
    },
    build: {
      sourcemap: false,
      commonjsOptions: {
        exclude: [/trueform/],
      },
    },
    optimizeDeps: {
      include: [
        '@vue/devtools-core',
        '@vue/devtools-kit',
        '@unhead/schema-org/vue',
        '@vueuse/core',
        '@unovis/vue',
        '@unovis/ts',
      ],
      exclude: ['@polydera/trueform'],
    },
    assetsInclude: ['**/*.wasm'],
  },

  llms: {
    domain: "https://trueform.polydera.com/",
    title: "trueform — Real-time geometric processing",
    description:
      "Geometry library for real-time spatial queries, mesh booleans, and topology. C++ header-only with Python bindings.",
    sections: [
      {
        title: "C++ API and guides",
        description:
          "C++ header-only API reference, usage guides, and examples for geometry processing.",
        contentCollection: "docsCpp",
      },
      {
        title: "Python API and guides",
        description:
          "Python bindings API reference, usage guides, and examples for geometry processing.",
        contentCollection: "docsPy",
      },
      {
        title: "TypeScript API and guides",
        description:
          "TypeScript SDK API reference, usage guides, and examples for geometry processing in the browser and Node.js.",
        contentCollection: "docsTs",
      },
    ],
  },
});

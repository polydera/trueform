// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
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
  ],

  devtools: {
    enabled: false,
  },

  css: ["~/assets/css/main.css"],

  content: {
    build: {
      markdown: {
        toc: {
          searchDepth: 1,
        },
        highlight: {
          langs: ["cpp", "python", "bash", "cmake"],
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

  // Sitemap configuration
  sitemap: {
    autoLastmod: true,
    zeroRuntime: true,
  },

  nitro: {
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
          exclude: ["/cpp/*", "/py/*", "/live-examples/*"],
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
    build: {
      sourcemap: false,
    },
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
    ],
  },
});

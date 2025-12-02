// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  modules: [
    "@nuxt/eslint",
    "@nuxt/image",
    "@nuxt/ui",
    "@nuxt/content",
    "nuxt-og-image",
    "./modules/copy-files",
    // "nuxt-llms",
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
    // Update this with your production domain
    // For Cloudflare Pages, this might be something like: https://trueform.pages.dev
    // You can also use environment variables: process.env.NUXT_PUBLIC_SITE_URL
    url: process.env.NUXT_PUBLIC_SITE_URL || "https://trueform.pages.dev",
    name: "trueform",
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
      sourcemap: false
    },
  },

  // llms: {
  //   domain: 'https://docs-template.nuxt.dev/',
  //   title: 'Nuxt Docs Template',
  //   description: 'A template for building documentation with Nuxt UI and Nuxt Content.',
  //   full: {
  //     title: 'Nuxt Docs Template - Full Documentation',
  //     description: 'This is the full documentation for the Nuxt Docs Template.'
  //   },
  //   sections: [
  //     {
  //       title: 'Getting Started',
  //       contentCollection: 'docs',
  //       contentFilters: [
  //         { field: 'path', operator: 'LIKE', value: '/getting-started%' }
  //       ]
  //     },
  //     {
  //       title: 'Essentials',
  //       contentCollection: 'docs',
  //       contentFilters: [
  //         { field: 'path', operator: 'LIKE', value: '/essentials%' }
  //       ]
  //     }
  //   ]
  // }
});

export default defineAppConfig({
  ui: {
    colors: {
      primary: "teal",
      neutral: "neutral",
    },
    footer: {
      slots: {
        root: "border-t border-default",
        left: "text-sm text-muted",
      },
    },
    prose: {
      codeIcon: {
        txt: "i-vscode-icons:file-type-cmake",
        hpp: "i-material-icon-theme:hpp"
      },
    },
    button: {
      slots: {
        base: 'cursor-pointer',
      },
    }
  },
  seo: {
    siteName: "trueform",
  },
  header: {
    title: "",
    to: "/",
    logo: {
      alt: "",
      light: "",
      dark: "",
    },
    search: true,
    colorMode: true,
    links: [
      {
        icon: "i-simple-icons-github",
        to: "https://github.com/polydera/trueform",
        target: "_blank",
        "aria-label": "GitHub",
      },
    ],
  },
  footer: {
    credits: `XLAB Medical • © ${new Date().getFullYear()}`,
    colorMode: false,
    // links: [{
    //   'icon': 'i-simple-icons-discord',
    //   'to': 'https://go.nuxt.com/discord',
    //   'target': '_blank',
    //   'aria-label': 'Nuxt on Discord'
    // }, {
    //   'icon': 'i-simple-icons-x',
    //   'to': 'https://go.nuxt.com/x',
    //   'target': '_blank',
    //   'aria-label': 'Nuxt on X'
    // }, {
    //   'icon': 'i-simple-icons-github',
    //   'to': 'https://github.com/nuxt/ui',
    //   'target': '_blank',
    //   'aria-label': 'Nuxt UI on GitHub'
    // }]
  },
  toc: {
    title: "Table of Contents",
    // bottom: {
    //   title: "Community",
    //   links: [
    //     {
    //       icon: "i-lucide-star",
    //       label: "Star on GitHub",
    //       to: "https://github.com/polydera/trueform",
    //       target: "_blank",
    //     },
    //   ],
    // },
  },
});

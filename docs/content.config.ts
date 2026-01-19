import { defineContentConfig, defineCollection, z } from '@nuxt/content'
import { asSitemapCollection } from '@nuxtjs/sitemap/content'

export default defineContentConfig({
  collections: {
    landing: defineCollection(asSitemapCollection({
        type: 'page',
        source: 'index.md'
      })),
    docsCpp: defineCollection(asSitemapCollection({
      type: 'page',
      source: {
        include: 'cpp/**',
      },
      schema: z.object({
        links: z.array(z.object({
          label: z.string(),
          icon: z.string(),
          to: z.string(),
          target: z.string().optional()
        })).optional()
      })
    })),
    docsPy: defineCollection(asSitemapCollection({
      type: 'page',
      source: {
        include: 'py/**',
      },
      schema: z.object({
        links: z.array(z.object({
          label: z.string(),
          icon: z.string(),
          to: z.string(),
          target: z.string().optional()
        })).optional()
      })
    })),
  }
})

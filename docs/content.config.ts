import { defineContentConfig, defineCollection, z } from '@nuxt/content'
import { defineSitemapSchema } from '@nuxtjs/sitemap/content'

const linksSchema = z.array(z.object({
  label: z.string(),
  icon: z.string(),
  to: z.string(),
  target: z.string().optional()
})).optional()

export default defineContentConfig({
  collections: {
    landing: defineCollection({
      type: 'page',
      source: 'index.md',
      schema: z.object({
        sitemap: defineSitemapSchema()
      })
    }),
    docsCpp: defineCollection({
      type: 'page',
      source: {
        include: 'cpp/**',
      },
      schema: z.object({
        sitemap: defineSitemapSchema(),
        links: linksSchema
      })
    }),
    docsPy: defineCollection({
      type: 'page',
      source: {
        include: 'py/**',
      },
      schema: z.object({
        sitemap: defineSitemapSchema(),
        links: linksSchema
      })
    }),
    docsTs: defineCollection({
      type: 'page',
      source: {
        include: 'ts/**',
      },
      schema: z.object({
        sitemap: defineSitemapSchema(),
        links: linksSchema
      })
    }),
  }
})

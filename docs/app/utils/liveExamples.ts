export interface ExampleMetadata {
  title: string
  description: string
  to: string
}

export const liveExamples: ExampleMetadata[] = [
  {
    title: "Boolean",
    description:
      "Visualizes real-time Boolean combinations between two `tf::forms` backed by `tf::tree` hierarchies.",
    to: "boolean",
  },
  {
    title: "Isobands",
    description:
      "Shows the iso-band slices reported by running `tf::search` over a form as a plane sweeps through the mesh hierarchy.",
    to: "isobands",
  },
  {
    title: "Positioning",
    description:
      "`tf::neighbor_search` keeps pairs of metric points aligned so two forms maintain contact-quality positioning.",
    to: "positioning",
  },
  {
    title: "Scalar Field Intersections",
    description:
      "Highlights the intersection curves produced when `tf::search(form, primitive)` walks a scalar field plane through a form.",
    to: "scalar-field-intersections",
  },
  {
    title: "Collision",
    description:
      "Demonstrates pairwise `tf::search` across two forms for collision detection, exposing timing directly from the spatial hierarchy.",
    to: "collision",
  },
  {
    title: "Mesh Intersections",
    description:
      "Displays intersection curves computed via `tf::search(form0, form1, ...)` and related form queries as meshes move.",
    to: "forms-intersections",
  },
]

export function getExampleMetadata(slug: string): ExampleMetadata | undefined {
  return liveExamples.find((example) => example.to === slug)
}


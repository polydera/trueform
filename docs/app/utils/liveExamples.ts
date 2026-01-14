export interface ExampleMetadata {
  title: string
  description: string
  to: string
}

export const liveExamples: ExampleMetadata[] = [
  // Mesh operations
  {
    title: "Boolean",
    description: "Drag a mesh. The boolean updates in real time.",
    to: "boolean",
  },
  {
    title: "Slicing",
    description: "Scroll to move the plane. Cross-sections update live.",
    to: "slicing",
  },
  // Mesh analysis
  {
    title: "Shape Index",
    description: "Hover over the mesh. Local shape distribution updates live.",
    to: "shape-histogram",
  },
  // Spatial queries
  {
    title: "Collision",
    description: "Drag a mesh. Contact detection runs live as you move.",
    to: "collision",
  },
  {
    title: "Closest Points",
    description:
      "Drag a mesh and release. It snaps to the nearest point instantly.",
    to: "closest-points",
  },
]

export function getExampleMetadata(slug: string): ExampleMetadata | undefined {
  return liveExamples.find((example) => example.to === slug)
}


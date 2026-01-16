export interface ExampleMetadata {
  title: string
  description: string
  to: string
}

export const liveExamples: ExampleMetadata[] = [
  // Mesh operations
  {
    title: "Boolean",
    description: "Interactive mesh boolean. Drag to see the difference update in real time.",
    to: "boolean",
  },
  {
    title: "Slicing",
    description: "Interactive mesh slicing. Scroll to move the slicing planes through geometry.",
    to: "slicing",
  },
  // Mesh analysis
  {
    title: "Shape Index",
    description: "Curvature-based shape analysis. Hover to see local shape index histograms.",
    to: "shape-histogram",
  },
  // Spatial queries
  {
    title: "Collision",
    description: "Interactive collision detection. Drag to see contacts update live.",
    to: "collision",
  },
  {
    title: "Closest Points",
    description: "Nearest point queries between meshes. Drag and release to snap to the closest point.",
    to: "closest-points",
  },
]

export function getExampleMetadata(slug: string): ExampleMetadata | undefined {
  return liveExamples.find((example) => example.to === slug)
}


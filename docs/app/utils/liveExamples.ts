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
    title: "Mesh Registration",
    description: "Automatic mesh registration. Move and rotate the source mesh, then click Align.",
    to: "alignment",
  },
  {
    title: "Slicing",
    description: "Interactive mesh slicing. Scroll to move the slicing planes through geometry.",
    to: "slicing",
  },
  {
    title: "Cross-Section",
    description: "Interactive cross-section visualization. Scroll to move the cutting plane through the mesh.",
    to: "cross-section",
  },
  // Mesh analysis
  {
    title: "Shape Index",
    description: "Curvature-based shape analysis. Hover to see local shape index histograms.",
    to: "shape-histogram",
  },
  {
    title: "Free-Form Smoothing",
    description: "Interactive mesh smoothing. Drag to smooth vertices with incremental tree updates.",
    to: "free-form-smoothing",
  },
  // Spatial queries
  {
    title: "Collision",
    description: "Interactive collision detection. Drag to see contacts update live.",
    to: "collision",
  },
  {
    title: "Closest Points",
    description: "Spatial neighbor search. Drag to see the closest point pair update in real time.",
    to: "closest-points",
  },
]

export function getExampleMetadata(slug: string): ExampleMetadata | undefined {
  return liveExamples.find((example) => example.to === slug)
}


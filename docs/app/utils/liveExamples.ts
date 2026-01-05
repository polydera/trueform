export interface ExampleMetadata {
  title: string
  description: string
  to: string
}

export const liveExamples: ExampleMetadata[] = [
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
  {
    title: "Closest Points",
    description:
      "Drag a mesh and release. It snaps to the nearest point instantly.",
    to: "closest-points",
  },
  {
    title: "Collision",
    description: "Drag a mesh. Contact detection runs live as you move.",
    to: "collision",
  },
]

export function getExampleMetadata(slug: string): ExampleMetadata | undefined {
  return liveExamples.find((example) => example.to === slug)
}


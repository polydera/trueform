const meshSizeOptions = [
  { label: "50k", value: "50k", size: "2.9 MB" },
  { label: "125k", value: "125k", size: "7.1 MB" },
  { label: "250k", value: "250k", size: "12.2 MB" },
  { label: "500k", value: "500k", size: "24 MB" }
] as const;

type MeshSizeOption = (typeof meshSizeOptions)[number];
export type MeshSizeValue = MeshSizeOption["value"];

export function useMeshSelection() {
  const meshSize = useState<MeshSizeValue>("mesh-size", () => "125k");

  const meshFilename = computed(() => `dragon-${meshSize.value}.stl`);
  const meshUrl = computed(() => `/stl/${meshFilename.value}`);

  const formatPolygonLabel = (count: number) => (count > 1 ? `${count} x ${meshSize.value}` : meshSize.value);

  const buildMeshes = (count: number) =>
    Array.from({ length: count }, (_value, index) => ({
      url: meshUrl.value,
      filename: count === 1 ? meshFilename.value : `dragon-${meshSize.value}-${index + 1}.stl`,
    }));

  return {
    meshSize,
    meshFilename,
    meshUrl,
    meshSizeOptions,
    formatPolygonLabel,
    buildMeshes,
  };
}

import type { MainModule } from "@/examples/native";
import * as THREE from "three";
import { ThreejsBase } from "@/examples/ThreejsBase";
import { fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";

export const SHAPE_HISTOGRAM_NUM_BINS = 10;

export class ShapeHistogramExample extends ThreejsBase {
  private vertexColorAttribute: THREE.BufferAttribute | null = null;
  public onHistogramUpdate?: (bins: number[]) => void;

  constructor(
    wasmInstance: MainModule,
    paths: string[],
    container: HTMLElement,
    isDarkMode = true,
  ) {
    // Pass showStats = false to hide GPU/CPU charts
    super(wasmInstance, paths, container, undefined, true, false, isDarkMode);

    this.lockRotationOnTouchDrag = true;

    // Setup vertex colors on the mesh geometry
    this.setupVertexColors();

    this.updateMeshes();
    fitCameraToAllMeshesFromZPlane(this.sceneBundle1, 1.5);
  }

  private setupVertexColors() {
    // Get the first instanced mesh geometry and add vertex colors
    const instancedMesh = this.instancedMeshes.get(0);
    if (!instancedMesh) return;

    const geometry = instancedMesh.geometry;
    const positionAttr = geometry.getAttribute("position");
    if (!positionAttr) return;

    const numVertices = positionAttr.count;

    // Create vertex color attribute (RGB, normalized 0-1)
    const colors = new Float32Array(numVertices * 3);
    for (let i = 0; i < numVertices * 3; i++) {
      colors[i] = 1.0; // Default white
    }
    this.vertexColorAttribute = new THREE.BufferAttribute(colors, 3);
    geometry.setAttribute("color", this.vertexColorAttribute);

    // Update material to use vertex colors
    const material = instancedMesh.material as THREE.MeshMatcapMaterial;
    material.vertexColors = true;
    material.needsUpdate = true;
  }

  public override updateMeshes() {
    super.updateMeshes();

    // Check if colors were updated
    const updated = this.wasmInstance.shape_histogram_colors_updated();
    if (updated) {
      this.updateVertexColors();
    }
    // Always try to update histogram to debug
    this.updateHistogram();
  }

  private updateVertexColors() {
    if (!this.vertexColorAttribute) return;

    const colorsView = this.wasmInstance.shape_histogram_get_vertex_colors();
    if (!colorsView) return;

    // Copy from WASM buffer (unsigned char RGB) to Three.js (float RGB)
    const colors = this.vertexColorAttribute.array as Float32Array;
    const numValues = Math.min(colors.length, colorsView.length);
    for (let i = 0; i < numValues; i++) {
      colors[i] = colorsView[i] / 255.0;
    }
    this.vertexColorAttribute.needsUpdate = true;
  }

  private updateHistogram() {
    const binsView = this.wasmInstance.shape_histogram_get_histogram_bins();
    if (!binsView) {
      return;
    }

    if (this.onHistogramUpdate) {
      const bins = Array.from(binsView.slice(0, SHAPE_HISTOGRAM_NUM_BINS));
      this.onHistogramUpdate(bins);
    }
  }

  public runMain() {
    this.wasmInstance.run_main_shape_histogram(this.paths[0]!);
    this.wasmInstance.FS.unlink(this.paths[0]);
  }

  public getAabbDiagonal(): number {
    return this.wasmInstance.shape_histogram_get_aabb_diagonal();
  }

  public setRadius(radius: number): void {
    this.wasmInstance.shape_histogram_set_radius(radius);
  }

  public override dispose() {
    super.dispose();
  }
}

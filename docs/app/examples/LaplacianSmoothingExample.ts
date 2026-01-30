import type { MainModule } from "@/examples/native";
import * as THREE from "three";
import { ThreejsBase } from "@/examples/ThreejsBase";
import { fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";

export class LaplacianSmoothingExample extends ThreejsBase {
  private vertexColorAttribute: THREE.BufferAttribute | null = null;
  private positionAttribute: THREE.BufferAttribute | null = null;

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

    this.positionAttribute = positionAttr as THREE.BufferAttribute;
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
    const colorsUpdated = this.wasmInstance.laplacian_smoothing_colors_updated();
    if (colorsUpdated) {
      this.updateVertexColors();
    }

    // Check if points were updated
    const pointsUpdated = this.wasmInstance.laplacian_smoothing_points_updated();
    if (pointsUpdated) {
      this.updateVertexPositions();
    }
  }

  private updateVertexColors() {
    if (!this.vertexColorAttribute) return;

    const colorsView = this.wasmInstance.laplacian_smoothing_get_vertex_colors();
    if (!colorsView) return;

    // Copy from WASM buffer (unsigned char RGB) to Three.js (float RGB)
    const colors = this.vertexColorAttribute.array as Float32Array;
    const numValues = Math.min(colors.length, colorsView.length);
    for (let i = 0; i < numValues; i++) {
      colors[i] = colorsView[i] / 255.0;
    }
    this.vertexColorAttribute.needsUpdate = true;
  }

  private updateVertexPositions() {
    if (!this.positionAttribute) return;

    const pointsView = this.wasmInstance.laplacian_smoothing_get_points();
    if (!pointsView) return;

    // Copy from WASM buffer to Three.js
    const positions = this.positionAttribute.array as Float32Array;
    const numValues = Math.min(positions.length, pointsView.length);
    for (let i = 0; i < numValues; i++) {
      positions[i] = pointsView[i];
    }
    this.positionAttribute.needsUpdate = true;

    // Update bounding sphere for proper rendering
    const instancedMesh = this.instancedMeshes.get(0);
    if (instancedMesh) {
      instancedMesh.geometry.computeBoundingSphere();
    }
  }

  public runMain() {
    this.wasmInstance.run_main_laplacian_smoothing(this.paths[0]!);
    this.wasmInstance.FS.unlink(this.paths[0]);
  }

  public getAabbDiagonal(): number {
    return this.wasmInstance.laplacian_smoothing_get_aabb_diagonal();
  }

  public setRadius(radius: number): void {
    this.wasmInstance.laplacian_smoothing_set_radius(radius);
  }

  public setLambda(lambda: number): void {
    this.wasmInstance.laplacian_smoothing_set_lambda(lambda);
  }

  public override dispose() {
    super.dispose();
  }
}

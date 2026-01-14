import type { MainModule } from "@/examples/native";
import * as THREE from "three";
import { ThreejsBase } from "@/examples/ThreejsBase";
import { fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";

const NUM_BINS = 10;

export class ShapeHistogramExample extends ThreejsBase {
  private histogramCanvas: HTMLCanvasElement;
  private histogramCtx: CanvasRenderingContext2D;
  private vertexColorAttribute: THREE.BufferAttribute | null = null;
  private container: HTMLElement;

  constructor(
    wasmInstance: MainModule,
    paths: string[],
    container: HTMLElement,
    isDarkMode = true,
  ) {
    // Pass showStats = false to hide GPU/CPU charts
    super(wasmInstance, paths, container, undefined, true, false, isDarkMode);
    this.container = container;

    // Setup histogram canvas overlay
    this.setupHistogramCanvas();

    // Setup vertex colors on the mesh geometry
    this.setupVertexColors();

    this.updateMeshes();
    fitCameraToAllMeshesFromZPlane(this.sceneBundle1, 1.5);
  }

  private setupHistogramCanvas() {
    // Ensure container has relative positioning for absolute child
    this.container.style.position = "relative";

    // Create canvas for histogram overlay
    this.histogramCanvas = document.createElement("canvas");
    this.histogramCanvas.style.position = "absolute";
    this.histogramCanvas.style.bottom = "12px";
    this.histogramCanvas.style.right = "12px";
    this.histogramCanvas.style.borderRadius = "8px";
    this.histogramCanvas.style.backgroundColor = "rgb(20, 20, 24)";
    this.histogramCanvas.style.pointerEvents = "none";
    this.histogramCanvas.style.zIndex = "10";

    this.container.appendChild(this.histogramCanvas);

    const ctx = this.histogramCanvas.getContext("2d");
    if (!ctx) throw new Error("Could not get 2D context");
    this.histogramCtx = ctx;

    // Size based on container, then draw
    this.resizeHistogramCanvas();

    // Listen for resize
    const resizeObserver = new ResizeObserver(() => {
      this.resizeHistogramCanvas();
      this.drawHistogramFrame();
    });
    resizeObserver.observe(this.container);
    this.addCleanup(() => resizeObserver.disconnect());
  }

  private resizeHistogramCanvas() {
    const containerWidth = this.container.clientWidth;
    // Scale: 300x180 on desktop (>768px), smaller on mobile
    const scale = containerWidth < 500 ? 0.65 : containerWidth < 768 ? 0.8 : 1.0;
    this.histogramCanvas.width = Math.round(300 * scale);
    this.histogramCanvas.height = Math.round(180 * scale);
    this.drawHistogramFrame();
  }

  private clearHistogram() {
    this.drawHistogramFrame();
  }

  private drawHistogramFrame() {
    const ctx = this.histogramCtx;
    const w = this.histogramCanvas.width;
    const h = this.histogramCanvas.height;
    const scale = w / 300; // Base width is 300
    const padding = Math.round(16 * scale);
    const titleFontSize = Math.round(12 * scale);
    const labelFontSize = Math.round(11 * scale);

    // Clear background
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgb(20, 20, 24)";
    ctx.fillRect(0, 0, w, h);

    // Draw title
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.font = `bold ${titleFontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText("Shape Index", w / 2, Math.round(18 * scale));

    // Draw axis labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
    ctx.font = `${labelFontSize}px sans-serif`;
    ctx.textAlign = "left";
    ctx.fillText("-1", padding, h - Math.round(5 * scale));
    ctx.textAlign = "right";
    ctx.fillText("1", w - padding, h - Math.round(5 * scale));
    ctx.textAlign = "center";
    ctx.fillText("0", w / 2, h - Math.round(5 * scale));
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
    // Always draw the frame first (title + axis labels)
    this.drawHistogramFrame();

    const binsView = this.wasmInstance.shape_histogram_get_histogram_bins();
    if (!binsView) {
      return;
    }

    // Find max for normalization
    let maxBin = 1;
    let totalCount = 0;
    for (let i = 0; i < NUM_BINS; i++) {
      if (binsView[i] > maxBin) maxBin = binsView[i];
      totalCount += binsView[i];
    }

    if (totalCount === 0) {
      return;
    }

    // Draw bars
    const ctx = this.histogramCtx;
    const w = this.histogramCanvas.width;
    const h = this.histogramCanvas.height;
    const scale = w / 300; // Base width is 300
    const padding = Math.round(16 * scale);
    const titleHeight = Math.round(26 * scale);
    const labelHeight = Math.round(20 * scale);
    const barAreaWidth = w - padding * 2;
    const barAreaHeight = h - titleHeight - labelHeight;
    const barWidth = barAreaWidth / NUM_BINS - Math.round(3 * scale);

    ctx.fillStyle = "#ff9977"; // Coral/peach color

    for (let i = 0; i < NUM_BINS; i++) {
      const barHeight = (binsView[i] / maxBin) * barAreaHeight;
      const x = padding + i * (barAreaWidth / NUM_BINS) + Math.round(1.5 * scale);
      const y = titleHeight + barAreaHeight - barHeight;

      ctx.fillRect(x, y, barWidth, barHeight);
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
    // Remove histogram canvas
    if (this.histogramCanvas && this.histogramCanvas.parentNode) {
      this.histogramCanvas.parentNode.removeChild(this.histogramCanvas);
    }
    super.dispose();
  }
}

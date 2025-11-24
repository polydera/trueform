import type { MainModule } from "@/examples/native";
import { ThreejsBase } from "@/examples/ThreejsBase";

export class CollisionExample extends ThreejsBase {
  constructor(wasmInstance: MainModule, paths: string[], container: HTMLElement, isDarkMode = true) {
    super(wasmInstance, paths, container, undefined, false, false, isDarkMode);
  }

  runMain() {
    const v = new this.wasmInstance.VectorString();
    for (const path of this.paths) {
      v.push_back(path);
    }
    this.wasmInstance.run_main_collisions(v);
    for (const path of this.paths) {
      this.wasmInstance.FS.unlink(path);
    }
  }
}

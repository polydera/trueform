import { MainModule } from "./webAssembly/build/dist/native.js";
import { TestClassThreejsBase } from "@/TestThreejsBase";

export class CollisionExample extends TestClassThreejsBase {
  constructor(
    wasmInstance: MainModule,
    paths: string[],
    container: HTMLElement
  ) {
    super(wasmInstance, paths, container);
  }

  runMain() {
    console.log("CollisionExample runMain");
    const v = new this.wasmInstance.VectorString();
    for (let i = 0; i < this.paths.length; i++) {
      v.push_back(this.paths[i]);
    }
    console.log("CollisionExample runMain v", v, v.size());
    this.wasmInstance.run_main_collisions(v);
    for (let i = 0; i < this.paths.length; i++) {
      this.wasmInstance.FS.unlink(this.paths[i]);
    }
  }
}

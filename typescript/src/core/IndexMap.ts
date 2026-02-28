/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */

import { NDArray, NDArrayInt32 } from "../ndarray/NDArray";

export interface NativeIndexMap {
  f: any;
  keptIds: any;
}

/**
 * Maps old indices to new indices after cleaning/reindexing.
 *
 * Returned by operations that remove or deduplicate elements.
 * Use `f` to remap old IDs and `keptIds` to extract surviving elements.
 */
export class IndexMap {
  /** Forward map: f[oldId] = newId (size = original count). */
  readonly f: NDArrayInt32;
  /** Original IDs that were retained (size = kept count). */
  readonly keptIds: NDArrayInt32;

  /** @internal */
  constructor(raw: NativeIndexMap) {
    this.f = new NDArray(raw.f, "int32");
    this.keptIds = new NDArray(raw.keptIds, "int32");
  }

  /** Release WASM memory for both arrays. */
  delete(): void {
    this.f.delete();
    this.keptIds.delete();
  }
}

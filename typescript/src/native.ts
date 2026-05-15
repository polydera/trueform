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

import { AsyncDispatcher } from "./internal/AsyncDispatcher";

let _native: any = null;
let _dispatcher: AsyncDispatcher | null = null;
let _initPromise: Promise<void> | null = null;

/**
 * Initialize the WASM module. Must be called (and awaited) once before
 * using any WASM-backed function. Subsequent calls return immediately.
 */
export async function init(): Promise<void> {
  if (!_initPromise) {
    _initPromise = (async () => {
      const createModule = (await import("./trueform_wasm.js"))
        .default;
      const m = await createModule();
      _native = m;
      _dispatcher = new AsyncDispatcher(m.wasmMemory, m.retrieve);
      // Warm up TBB thread pool via async round-trip
      await _dispatcher.run(() => m.init_tbb());
    })();
  }
  return _initPromise;
}

/** Returns the raw embind module. */
export function native(): any {
  if (!_native) throw new Error("Call tf.init() before using WASM functions.");
  return _native;
}

/** Returns the async dispatcher. */
export function dispatcher(): AsyncDispatcher {
  if (!_dispatcher)
    throw new Error("Call tf.init() before using WASM functions.");
  return _dispatcher;
}

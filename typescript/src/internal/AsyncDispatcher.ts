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

/**
 * Internal async dispatch wrapper.
 *
 * Bridges C++ TBB dispatch with JS Promises via Atomics.waitAsync.
 * Each operation does two WASM calls: dispatch (alloc + run) and retrieve (get + free).
 * Resolution happens as a microtask (not macrotask), so latency is ~0.
 *
 * The C++ side stores a type-erased converter per context, so one generic
 * `retrieve` embind function serves all return types.
 */
interface NodeProcessLike {
  versions?: { node?: string };
}
interface GlobalWithProcess {
  process?: NodeProcessLike;
}
const _isNode = typeof (globalThis as GlobalWithProcess).process !== "undefined" &&
  (globalThis as GlobalWithProcess).process?.versions?.node != null;

export class AsyncDispatcher {
  private _memory: WebAssembly.Memory;
  private _retrieve: (slot: number) => any;

  constructor(memory: WebAssembly.Memory, retrieve: (slot: number) => any) {
    this._memory = memory;
    this._retrieve = retrieve;
  }

  /**
   * Dispatches an async operation and returns a Promise for the result.
   *
   * @param dispatch - Calls the native dispatch_* function, returns slot address.
   * @param unwrap - Optional transform on the raw JS value returned by retrieve.
   */
  async run<T>(
    dispatch: () => number,
    unwrap?: (raw: any) => T,
  ): Promise<T> {
    const slot = dispatch();

    // Create Int32Array view at the status field address in SharedArrayBuffer
    const view = new Int32Array(this._memory.buffer, slot, 1);

    // Wait for status to change from 0 (pending)
    const result = Atomics.waitAsync(view, 0, 0);
    if (result.async) {
      if (_isNode) {
        // Node.js: Atomics.waitAsync doesn't ref the event loop —
        // without a ref'd handle, Node exits before the worker can notify.
        const keepalive = setInterval(() => {}, 1 << 30);
        try { await result.value; } finally { clearInterval(keepalive); }
      } else {
        await result.value;
      }
    }

    // Check for error
    const status = Atomics.load(view, 0);
    if (status === -1) {
      throw new Error("Async operation failed");
    }

    // Single generic retrieve — C++ converter handles the type
    const raw = this._retrieve(slot);
    return unwrap ? unwrap(raw) : raw;
  }

  /**
   * Like run(), but polls with Atomics.load + setTimeout instead of
   * Atomics.waitAsync. Works in environments where waitAsync is unavailable
   * (e.g. Node.js). Used for thread pool initialization.
   */
  async poll<T>(
    dispatch: () => number,
    unwrap?: (raw: any) => T,
  ): Promise<T> {
    const slot = dispatch();
    const view = new Int32Array(this._memory.buffer, slot, 1);

    while (Atomics.load(view, 0) === 0) {
      await new Promise<void>(r => setTimeout(r, 0));
    }

    const status = Atomics.load(view, 0);
    if (status === -1) {
      throw new Error("Async operation failed");
    }

    const raw = this._retrieve(slot);
    return unwrap ? unwrap(raw) : raw;
  }
}

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
import { releaseHandle } from "./registry";

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
      this._retrieve(slot);
      throw new Error("Async operation failed");
    }

    // Single generic retrieve — C++ converter handles the type
    return this._convert(slot, unwrap);
  }

  /**
   * Retrieve the result and convert it. When `unwrap` throws, the native
   * handles of the result that were not yet taken over by a wrapper (and
   * the ones that were: the registry tolerates an already deleted handle)
   * are released here, so a failed conversion leaves nothing pinned.
   */
  private _convert<T>(slot: number, unwrap?: (raw: any) => T): T {
    const raw = this._retrieve(slot);
    if (!unwrap) return raw;
    try {
      return unwrap(raw);
    } catch (e) {
      const release = (v: any) => {
        if (v !== null && typeof v === "object" && typeof v.delete === "function") releaseHandle(v);
      };
      release(raw);
      if (raw !== null && typeof raw === "object") Object.values(raw).forEach(release);
      throw e;
    }
  }

  /**
   * Like run(), but polls with Atomics.load + setTimeout instead of
   * Atomics.waitAsync. Available as a fallback for environments where
   * waitAsync cannot be used.
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
      this._retrieve(slot);
      throw new Error("Async operation failed");
    }

    return this._convert(slot, unwrap);
  }
}

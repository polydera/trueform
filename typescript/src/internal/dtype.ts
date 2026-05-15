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

/** Throws if any input's dtype differs from the first. */
export function assertSameDtype(
  inputs: ReadonlyArray<{ dtype: string }>,
  names?: ReadonlyArray<string>,
): void {
  const first = inputs[0].dtype;
  for (let i = 1; i < inputs.length; i++) {
    if (inputs[i].dtype !== first) {
      const a = names?.[0] ?? "input 0";
      const b = names?.[i] ?? `input ${i}`;
      throw new Error(
        `dtype mismatch: ${a}=${first}, ${b}=${inputs[i].dtype}`,
      );
    }
  }
}

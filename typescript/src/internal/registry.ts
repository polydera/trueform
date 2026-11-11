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

interface Release {
  handle: { delete(): void; isDeleted?(): boolean };
}

/** Delete an embind handle once; a handle already deleted is left alone. */
export function releaseHandle(handle: Release["handle"]): void {
  if (!handle.isDeleted?.()) handle.delete();
}

export const registry = new FinalizationRegistry<Release>((ref) => {
  releaseHandle(ref.handle);
});

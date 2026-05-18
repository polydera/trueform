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

export async function toBlobURL(url: string, mime: string): Promise<string> {
  const buf = await fetch(url).then((r) => r.arrayBuffer());
  return URL.createObjectURL(new Blob([buf], { type: mime }));
}

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

// Auto-init entry. Imports the SDK from ./manual (the source of truth for the
// export surface) and triggers init() at module load. Consumers that need
// to control where the wasm/worker are loaded from should import from
// "@polydera/trueform/manual" instead and call init({wasmUrl, workerUrl})
// themselves.

import { init } from "./native";
await init();
export * from "./manual";

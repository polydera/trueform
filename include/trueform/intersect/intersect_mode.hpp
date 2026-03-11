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
#pragma once

namespace tf {

enum class intersect_mode {
  sos,        // SoS fan triangulation — all records are (edge, face)
  primitives  // Conforming 5-type classification (EF, EE, VE, VF, VV)
};

} // namespace tf

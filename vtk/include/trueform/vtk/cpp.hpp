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

/// The tier that reads a vtkPolyData as a trueform mesh. It stands on the
/// compiled C++ facade rather than on the header-only library, which is why
/// `trueform/vtk.hpp` does not export it: a caller that has `tf::trueform_cpp`
/// includes this, and one that does not has nothing to include.

#include "./cpp/mesh_source.hpp" // IWYU pragma: export

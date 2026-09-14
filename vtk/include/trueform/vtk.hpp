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

/// The `cpp` tier -- `trueform/vtk/cpp.hpp` -- is not exported here: it stands
/// on the compiled C++ facade rather than on the header-only library, and this
/// umbrella states what `tf::vtk` itself links against.

#include "./vtk/core.hpp"      // IWYU pragma: export
#include "./vtk/filters.hpp"   // IWYU pragma: export
#include "./vtk/functions.hpp" // IWYU pragma: export

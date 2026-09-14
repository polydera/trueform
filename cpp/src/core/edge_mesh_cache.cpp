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
#include "./edge_mesh_cache_impl.hpp"

#include "trueform/cpp/core/matrix.hpp"

namespace tf::cpp {
#define TF_CPP_INSTANTIATE_EDGE_MESH_CACHE(Index, Real, Dims)                  \
  template class edge_mesh_cache<Index, Real, Dims>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS(TF_CPP_INSTANTIATE_EDGE_MESH_CACHE)

#undef TF_CPP_INSTANTIATE_EDGE_MESH_CACHE
} // namespace tf::cpp

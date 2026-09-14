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
#include "./cache_impl.hpp"

namespace tf::cpp {
#define TF_CPP_INSTANTIATE_CACHE(Index, Real, Dims, Ngon)                      \
  template class cache<Index, Real, Dims, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_INSTANTIATE_CACHE)

#undef TF_CPP_INSTANTIATE_CACHE
} // namespace tf::cpp

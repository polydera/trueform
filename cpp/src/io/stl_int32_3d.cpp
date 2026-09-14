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
#include "read_stl_impl.hpp"
#include "stl_instantiations.hpp"
#include "write_stl_impl.hpp"

#include <cstdint>

namespace tf::cpp {
#if TF_CPP_MATRIX_HAS_FLOAT
TF_CPP_INSTANTIATE_READ_STL(std::int32_t);
#endif
#define TF_CPP_INSTANTIATE_WRITE_STL_AT_REAL(Real)                             \
  TF_CPP_INSTANTIATE_WRITE_STL(std::int32_t, Real)

TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_INSTANTIATE_WRITE_STL_AT_REAL)

#undef TF_CPP_INSTANTIATE_WRITE_STL_AT_REAL
} // namespace tf::cpp

#undef TF_CPP_INSTANTIATE_WRITE_STL
#undef TF_CPP_INSTANTIATE_READ_STL

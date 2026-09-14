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
#include "obj_instantiations.hpp"
#include "read_obj_impl.hpp"
#include "write_obj_impl.hpp"

#include <cstdint>

namespace tf::cpp {
#define TF_CPP_INSTANTIATE_OBJ_IO_AT_ARITY(Ngon)                               \
  TF_CPP_INSTANTIATE_OBJ_IO(std::int32_t, double, Ngon)

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_OBJ_IO_AT_ARITY)

#undef TF_CPP_INSTANTIATE_OBJ_IO_AT_ARITY
} // namespace tf::cpp

#undef TF_CPP_INSTANTIATE_OBJ_IO

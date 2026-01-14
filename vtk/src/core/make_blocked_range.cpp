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
#include <trueform/vtk/core/make_blocked_range.hpp>

namespace tf::vtk {

auto make_blocked_range(vtkFloatArray *array, std::size_t block_size)
    -> float_dynamic_blocked_range_t {
  return tf::make_blocked_range(make_range(array), block_size);
}

auto make_blocked_range(vtkDoubleArray *array, std::size_t block_size)
    -> double_dynamic_blocked_range_t {
  return tf::make_blocked_range(make_range(array), block_size);
}

auto make_blocked_range(vtkIntArray *array, std::size_t block_size)
    -> int_dynamic_blocked_range_t {
  return tf::make_blocked_range(make_range(array), block_size);
}

auto make_blocked_range(vtkIdTypeArray *array, std::size_t block_size)
    -> vtkIdType_dynamic_blocked_range_t {
  return tf::make_blocked_range(make_range(array), block_size);
}

auto make_blocked_range(vtkSignedCharArray *array, std::size_t block_size)
    -> signed_char_dynamic_blocked_range_t {
  return tf::make_blocked_range(make_range(array), block_size);
}

auto make_blocked_range(vtkUnsignedCharArray *array, std::size_t block_size)
    -> unsigned_char_dynamic_blocked_range_t {
  return tf::make_blocked_range(make_range(array), block_size);
}

} // namespace tf::vtk

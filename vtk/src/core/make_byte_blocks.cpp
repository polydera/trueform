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
#include <trueform/vtk/core/make_byte_blocks.hpp>
#include <vtkDataArray.h>

namespace tf::vtk {

auto make_byte_blocks(vtkDataArray *array) -> byte_blocks_t {
  if (!array || array->GetNumberOfTuples() == 0) {
    return tf::make_blocked_range(
        tf::make_range(static_cast<unsigned char *>(nullptr), std::size_t{0}),
        std::size_t{1});
  }
  auto *ptr = static_cast<unsigned char *>(array->GetVoidPointer(0));
  auto tuple_size =
      static_cast<std::size_t>(array->GetNumberOfComponents()) *
      static_cast<std::size_t>(array->GetDataTypeSize());
  auto total_bytes =
      static_cast<std::size_t>(array->GetNumberOfTuples()) * tuple_size;
  return tf::make_blocked_range(tf::make_range(ptr, total_bytes), tuple_size);
}

} // namespace tf::vtk

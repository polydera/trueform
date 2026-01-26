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
#include <trueform/core.hpp>
#include <cstddef>

class vtkDataArray;

namespace tf::vtk {

/// Blocked range of bytes, where each block is one tuple
using byte_blocks_t =
    decltype(tf::make_blocked_range(std::declval<tf::range<unsigned char *, tf::dynamic_size>>(),
                                    std::size_t{}));

/// @brief Creates a blocked range view over vtkDataArray tuple bytes.
/// Each block represents one tuple as raw bytes (numComponents * elementSize).
/// @param array The vtkDataArray to view.
/// @return A blocked range where each element is the bytes of one tuple.
auto make_byte_blocks(vtkDataArray *array) -> byte_blocks_t;

} // namespace tf::vtk

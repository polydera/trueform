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
#include <vtkType.h>

class vtkFloatArray;
class vtkDoubleArray;
class vtkIntArray;
class vtkIdTypeArray;
class vtkSignedCharArray;
class vtkUnsignedCharArray;

namespace tf::vtk {

// Range types
using float_range_t = tf::range<float *, tf::dynamic_size>;
using double_range_t = tf::range<double *, tf::dynamic_size>;
using int_range_t = tf::range<int *, tf::dynamic_size>;
using vtkIdType_range_t = tf::range<vtkIdType *, tf::dynamic_size>;
using signed_char_range_t = tf::range<signed char *, tf::dynamic_size>;
using unsigned_char_range_t = tf::range<unsigned char *, tf::dynamic_size>;

/// @brief Creates a range view over vtkFloatArray data (zero-copy).
auto make_range(vtkFloatArray *array) -> float_range_t;

/// @brief Creates a range view over vtkDoubleArray data (zero-copy).
auto make_range(vtkDoubleArray *array) -> double_range_t;

/// @brief Creates a range view over vtkIntArray data (zero-copy).
auto make_range(vtkIntArray *array) -> int_range_t;

/// @brief Creates a range view over vtkIdTypeArray data (zero-copy).
auto make_range(vtkIdTypeArray *array) -> vtkIdType_range_t;

/// @brief Creates a range view over vtkSignedCharArray data (zero-copy).
auto make_range(vtkSignedCharArray *array) -> signed_char_range_t;

/// @brief Creates a range view over vtkUnsignedCharArray data (zero-copy).
auto make_range(vtkUnsignedCharArray *array) -> unsigned_char_range_t;

} // namespace tf::vtk

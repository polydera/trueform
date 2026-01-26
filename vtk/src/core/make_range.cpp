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
#include <trueform/vtk/core/make_range.hpp>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkSignedCharArray.h>
#include <vtkUnsignedCharArray.h>

namespace tf::vtk {

auto make_range(vtkFloatArray *array) -> float_range_t {
  if (!array)
    return tf::make_range(static_cast<float *>(nullptr), std::size_t{0});
  return tf::make_range(static_cast<float *>(array->GetPointer(0)),
                        array->GetNumberOfValues());
}

auto make_range(vtkDoubleArray *array) -> double_range_t {
  if (!array)
    return tf::make_range(static_cast<double *>(nullptr), std::size_t{0});
  return tf::make_range(static_cast<double *>(array->GetPointer(0)),
                        array->GetNumberOfValues());
}

auto make_range(vtkIntArray *array) -> int_range_t {
  if (!array)
    return tf::make_range(static_cast<int *>(nullptr), std::size_t{0});
  return tf::make_range(static_cast<int *>(array->GetPointer(0)),
                        array->GetNumberOfValues());
}

auto make_range(vtkIdTypeArray *array) -> vtkIdType_range_t {
  if (!array)
    return tf::make_range(static_cast<vtkIdType *>(nullptr), std::size_t{0});
  return tf::make_range(static_cast<vtkIdType *>(array->GetPointer(0)),
                        array->GetNumberOfValues());
}

auto make_range(vtkSignedCharArray *array) -> signed_char_range_t {
  if (!array)
    return tf::make_range(static_cast<signed char *>(nullptr), std::size_t{0});
  return tf::make_range(static_cast<signed char *>(array->GetPointer(0)),
                        array->GetNumberOfValues());
}

auto make_range(vtkUnsignedCharArray *array) -> unsigned_char_range_t {
  if (!array)
    return tf::make_range(static_cast<unsigned char *>(nullptr), std::size_t{0});
  return tf::make_range(static_cast<unsigned char *>(array->GetPointer(0)),
                        array->GetNumberOfValues());
}

} // namespace tf::vtk

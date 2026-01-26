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
#include <vtkSmartPointer.h>
#include <vtkType.h>

class vtkSignedCharArray;
class vtkIntArray;
class vtkIdTypeArray;
class vtkFloatArray;

namespace tf::vtk {

/// @brief Creates vtkSignedCharArray from a buffer (copies data).
/// @param buffer Trueform buffer.
/// @return A new vtkSignedCharArray object with copied data.
auto make_vtk_array(const tf::buffer<std::int8_t> &buffer)
    -> vtkSmartPointer<vtkSignedCharArray>;

/// @brief Creates vtkSignedCharArray from a buffer (moves data).
/// @param buffer Trueform buffer (consumed).
/// @return A new vtkSignedCharArray object with transferred ownership.
auto make_vtk_array(tf::buffer<std::int8_t> &&buffer)
    -> vtkSmartPointer<vtkSignedCharArray>;

/// @brief Creates vtkIntArray from a buffer (copies data).
/// @param buffer Trueform buffer.
/// @return A new vtkIntArray object with copied data.
auto make_vtk_array(const tf::buffer<int> &buffer)
    -> vtkSmartPointer<vtkIntArray>;

/// @brief Creates vtkIntArray from a buffer (moves data).
/// @param buffer Trueform buffer (consumed).
/// @return A new vtkIntArray object with transferred ownership.
auto make_vtk_array(tf::buffer<int> &&buffer)
    -> vtkSmartPointer<vtkIntArray>;

/// @brief Creates vtkIdTypeArray from a buffer (copies data).
/// @param buffer Trueform buffer.
/// @return A new vtkIdTypeArray object with copied data.
auto make_vtk_array(const tf::buffer<vtkIdType> &buffer)
    -> vtkSmartPointer<vtkIdTypeArray>;

/// @brief Creates vtkIdTypeArray from a buffer (moves data).
/// @param buffer Trueform buffer (consumed).
/// @return A new vtkIdTypeArray object with transferred ownership.
auto make_vtk_array(tf::buffer<vtkIdType> &&buffer)
    -> vtkSmartPointer<vtkIdTypeArray>;

/// @brief Creates vtkIdTypeArray from a buffer (copies data).
/// @param buffer Trueform buffer.
/// @return A new vtkFloatArray object with copied data.
auto make_vtk_array(const tf::buffer<float> &buffer)
    -> vtkSmartPointer<vtkFloatArray>;

/// @brief Creates vtkIdTypeArray from a buffer (moves data).
/// @param buffer Trueform buffer (consumed).
/// @return A new vtkFloatArray object with transferred ownership.
auto make_vtk_array(tf::buffer<float> &&buffer)
    -> vtkSmartPointer<vtkFloatArray>;

/// @brief Creates vtkFloatArray from a unit_vectors_buffer (copies data).
/// @param buffer Trueform unit_vectors_buffer.
/// @return A new vtkFloatArray with 3 components per tuple.
auto make_vtk_array(const tf::unit_vectors_buffer<float, 3> &buffer)
    -> vtkSmartPointer<vtkFloatArray>;

/// @brief Creates vtkFloatArray from a unit_vectors_buffer (moves data).
/// @param buffer Trueform unit_vectors_buffer (consumed).
/// @return A new vtkFloatArray with 3 components per tuple.
auto make_vtk_array(tf::unit_vectors_buffer<float, 3> &&buffer)
    -> vtkSmartPointer<vtkFloatArray>;

} // namespace tf::vtk

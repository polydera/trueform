/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/core.hpp>
#include <vtkSmartPointer.h>
#include <vtkType.h>

class vtkSignedCharArray;
class vtkIntArray;
class vtkIdTypeArray;

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

} // namespace tf::vtk

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/core/index_map.hpp>
#include <vtkSmartPointer.h>
#include <vtkType.h>

class vtkDataArray;

namespace tf::vtk {

/// @brief Creates a new vtkDataArray with tuples reindexed by the given map.
/// @param array The source array.
/// @param im The index map (kept_ids specifies which tuples to keep).
/// @return A new array of the same type with reindexed tuples.
auto make_vtk_array_reindexed(vtkDataArray *array,
                              const tf::index_map_buffer<vtkIdType> &im)
    -> vtkSmartPointer<vtkDataArray>;

} // namespace tf::vtk

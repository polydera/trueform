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
#include <trueform/vtk/core/make_vtk_array_reindexed.hpp>
#include <vtkDataArray.h>

namespace tf::vtk {

auto make_vtk_array_reindexed(vtkDataArray *array,
                              const tf::index_map_buffer<vtkIdType> &im)
    -> vtkSmartPointer<vtkDataArray> {
  if (!array) {
    return nullptr;
  }

  auto out = vtkSmartPointer<vtkDataArray>::Take(array->NewInstance());
  out->SetNumberOfComponents(array->GetNumberOfComponents());
  out->SetNumberOfTuples(static_cast<vtkIdType>(im.kept_ids().size()));
  out->SetName(array->GetName());

  auto src_blocks = make_byte_blocks(array);
  auto dst_blocks = make_byte_blocks(out);
  tf::parallel_copy_blocked(tf::make_indirect_range(im.kept_ids(), src_blocks),
                            dst_blocks);
  return out;
}

} // namespace tf::vtk

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/vtk/functions/read_stl.hpp>
#include <trueform/io/read_stl.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>

namespace tf::vtk {

auto read_stl(const std::string &file_name) -> vtkSmartPointer<polydata> {
  auto polys = tf::read_stl<vtkIdType>(file_name);
  auto result = vtkSmartPointer<polydata>::New();
  result->ShallowCopy(make_vtk_polydata(std::move(polys)));
  return result;
}

} // namespace tf::vtk

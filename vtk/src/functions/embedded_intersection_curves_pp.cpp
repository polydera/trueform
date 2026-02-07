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
#include "./impl/embedded_intersection_curves_impl.hpp"

namespace tf::vtk {

using namespace impl;

auto embedded_intersection_curves(polydata *input0, polydata *input1)
    -> vtkSmartPointer<polydata> {
  if (!input0 || !input1) {
    return nullptr;
  }

  return dispatch_eic(input0, input1, [](auto &&base0, auto &&base1) {
    return compute_embedded_intersection_curves(base0, base1);
  });
}

auto embedded_intersection_curves(polydata *input0, polydata *input1,
                                  tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>> {
  if (!input0 || !input1) {
    return {nullptr, nullptr};
  }

  return dispatch_eic(input0, input1, [](auto &&base0, auto &&base1) {
    return compute_embedded_intersection_curves_with_curves(base0, base1);
  });
}

} // namespace tf::vtk

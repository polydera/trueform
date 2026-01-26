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
#include "./impl/make_boolean_impl.hpp"

namespace tf::vtk {

using namespace impl;

auto make_boolean(polydata *input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op)
    -> vtkSmartPointer<polydata> {
  if (!input0 || !input1.first || !input1.second) {
    return nullptr;
  }

  return dispatch(
      input0, input1.first, [op, m1 = input1.second](auto &&base0, auto &&base1) {
        auto frame1 = make_frame(m1);
        return compute_boolean(base0, base1 | tf::tag(frame1), op);
      });
}

auto make_boolean(polydata *input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op, tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>> {
  if (!input0 || !input1.first || !input1.second) {
    return {nullptr, nullptr};
  }

  return dispatch(
      input0, input1.first, [op, m1 = input1.second](auto &&base0, auto &&base1) {
        auto frame1 = make_frame(m1);
        return compute_boolean_with_curves(base0, base1 | tf::tag(frame1), op);
      });
}

} // namespace tf::vtk

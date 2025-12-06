/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include "./impl/make_boolean_impl.hpp"

namespace tf::vtk {

using namespace impl;

auto make_boolean(std::pair<polydata *, vtkMatrix4x4 *> input0,
                  polydata *input1, tf::boolean_op op)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>> {
  if (!input0.first || !input1 || !input0.second) {
    return {nullptr, nullptr};
  }

  return dispatch(
      input0.first, input1, [op, m0 = input0.second](auto &&base0, auto &&base1) {
        auto frame0 = make_frame(m0);
        return compute_boolean(base0 | tf::tag(frame0), base1, op);
      });
}

auto make_boolean(std::pair<polydata *, vtkMatrix4x4 *> input0,
                  polydata *input1, tf::boolean_op op, tf::return_curves_t)
    -> std::tuple<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>,
                  vtkSmartPointer<polydata>> {
  if (!input0.first || !input1 || !input0.second) {
    return {nullptr, nullptr, nullptr};
  }

  return dispatch(
      input0.first, input1, [op, m0 = input0.second](auto &&base0, auto &&base1) {
        auto frame0 = make_frame(m0);
        return compute_boolean_with_curves(base0 | tf::tag(frame0), base1, op);
      });
}

} // namespace tf::vtk

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
#include <trueform/intersect/make_intersection_curves.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <trueform/vtk/functions/make_intersection_curves.hpp>

namespace tf::vtk {

namespace {

auto make_frame(vtkMatrix4x4 *matrix) -> tf::frame<double, 3> {
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  return frame;
}

auto make_base(polydata *in) {
  return in->polygons() | tf::tag(in->face_membership()) |
         tf::tag(in->manifold_edge_link()) | tf::tag(in->poly_tree());
}

template <typename F0, typename F1>
auto compute_curves(F0 &&form0, F1 &&form1) -> vtkSmartPointer<polydata> {
  auto curves = tf::make_intersection_curves(form0, form1);
  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(curves)));
  return out;
}

template <typename Base0, typename Base1>
auto run_without_transforms(Base0 &&base0, Base1 &&base1)
    -> vtkSmartPointer<polydata> {
  return compute_curves(base0, base1);
}

template <typename Base0, typename Base1>
auto run_with_transform0(Base0 &&base0, Base1 &&base1, vtkMatrix4x4 *matrix0)
    -> vtkSmartPointer<polydata> {
  auto frame0 = make_frame(matrix0);
  return compute_curves(base0 | tf::tag(frame0), base1);
}

template <typename Base0, typename Base1>
auto run_with_transform1(Base0 &&base0, Base1 &&base1, vtkMatrix4x4 *matrix1)
    -> vtkSmartPointer<polydata> {
  auto frame1 = make_frame(matrix1);
  return compute_curves(base0, base1 | tf::tag(frame1));
}

template <typename Base0, typename Base1>
auto run_with_both_transforms(Base0 &&base0, Base1 &&base1,
                              vtkMatrix4x4 *matrix0, vtkMatrix4x4 *matrix1)
    -> vtkSmartPointer<polydata> {
  auto frame0 = make_frame(matrix0);
  auto frame1 = make_frame(matrix1);
  return compute_curves(base0 | tf::tag(frame0), base1 | tf::tag(frame1));
}

template <typename Runner>
auto dispatch(polydata *in0, polydata *in1, Runner &&runner)
    -> vtkSmartPointer<polydata> {
  return runner(make_base(in0), make_base(in1));
}

} // namespace

auto make_intersection_curves(polydata *input0, polydata *input1)
    -> vtkSmartPointer<polydata> {
  if (!input0 || !input1) {
    return nullptr;
  }

  return dispatch(input0, input1, [](auto &&base0, auto &&base1) {
    return run_without_transforms(base0, base1);
  });
}

auto make_intersection_curves(std::pair<polydata *, vtkMatrix4x4 *> input0,
                              polydata *input1)
    -> vtkSmartPointer<polydata> {
  if (!input0.first || !input1 || !input0.second) {
    return nullptr;
  }

  return dispatch(input0.first, input1,
                  [m0 = input0.second](auto &&base0, auto &&base1) {
                    return run_with_transform0(base0, base1, m0);
                  });
}

auto make_intersection_curves(polydata *input0,
                              std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> vtkSmartPointer<polydata> {
  if (!input0 || !input1.first || !input1.second) {
    return nullptr;
  }

  return dispatch(input0, input1.first,
                  [m1 = input1.second](auto &&base0, auto &&base1) {
                    return run_with_transform1(base0, base1, m1);
                  });
}

auto make_intersection_curves(std::pair<polydata *, vtkMatrix4x4 *> input0,
                              std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> vtkSmartPointer<polydata> {
  if (!input0.first || !input1.first || !input0.second || !input1.second) {
    return nullptr;
  }

  return dispatch(
      input0.first, input1.first,
      [m0 = input0.second, m1 = input1.second](auto &&base0, auto &&base1) {
        return run_with_both_transforms(base0, base1, m0, m1);
      });
}

} // namespace tf::vtk

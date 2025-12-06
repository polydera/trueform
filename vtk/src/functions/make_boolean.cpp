/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/cut/make_boolean.hpp>
#include <trueform/vtk/core/make_vtk_array.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <trueform/vtk/functions/make_boolean.hpp>
#include <vtkCellData.h>
#include <vtkSignedCharArray.h>

namespace tf::vtk {

namespace {

auto make_frame(vtkMatrix4x4 *matrix) -> tf::frame<double, 3> {
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  return frame;
}

auto make_base_tri(polydata *in) {
  return tf::make_form(in->poly_tree(),
                       in->triangle_polygons() |
                           tf::tag(in->face_membership()) |
                           tf::tag(in->triangle_manifold_edge_link()));
}

auto make_base_dyn(polydata *in) {
  return tf::make_form(in->poly_tree(),
                       in->polygons() | tf::tag(in->face_membership()) |
                           tf::tag(in->manifold_edge_link()));
}

template <typename F0, typename F1>
auto compute_boolean(F0 &&form0, F1 &&form1, tf::boolean_op op)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>> {
  auto [mesh, labels] = tf::make_boolean(form0, form1, op);

  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(mesh)));

  auto label_array = make_vtk_array(std::move(labels));
  label_array->SetName("labels");
  out->GetCellData()->SetScalars(label_array);

  return {out, label_array};
}

template <typename F0, typename F1>
auto compute_boolean_with_curves(F0 &&form0, F1 &&form1, tf::boolean_op op)
    -> std::tuple<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>,
                  vtkSmartPointer<polydata>> {
  auto [mesh, labels, curves] =
      tf::make_boolean(form0, form1, op, tf::return_curves);

  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(mesh)));

  auto label_array = make_vtk_array(std::move(labels));
  label_array->SetName("labels");
  out->GetCellData()->SetScalars(label_array);

  auto out_curves = vtkSmartPointer<polydata>::New();
  out_curves->ShallowCopy(make_vtk_polydata(std::move(curves)));

  return {out, label_array, out_curves};
}

template <typename Runner>
auto dispatch(polydata *in0, polydata *in1, Runner &&runner) {
  if (in0->is_triangles() && in1->is_triangles()) {
    return runner(make_base_tri(in0), make_base_tri(in1));
  } else if (in0->is_triangles()) {
    return runner(make_base_tri(in0), make_base_dyn(in1));
  } else if (in1->is_triangles()) {
    return runner(make_base_dyn(in0), make_base_tri(in1));
  } else {
    return runner(make_base_dyn(in0), make_base_dyn(in1));
  }
}

} // namespace

// Without curves

auto make_boolean(polydata *input0, polydata *input1, tf::boolean_op op)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>> {
  if (!input0 || !input1) {
    return {nullptr, nullptr};
  }

  return dispatch(input0, input1, [op](auto &&base0, auto &&base1) {
    return compute_boolean(base0, base1, op);
  });
}

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

auto make_boolean(polydata *input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>> {
  if (!input0 || !input1.first || !input1.second) {
    return {nullptr, nullptr};
  }

  return dispatch(
      input0, input1.first, [op, m1 = input1.second](auto &&base0, auto &&base1) {
        auto frame1 = make_frame(m1);
        return compute_boolean(base0, base1 | tf::tag(frame1), op);
      });
}

auto make_boolean(std::pair<polydata *, vtkMatrix4x4 *> input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>> {
  if (!input0.first || !input1.first || !input0.second || !input1.second) {
    return {nullptr, nullptr};
  }

  return dispatch(
      input0.first, input1.first,
      [op, m0 = input0.second, m1 = input1.second](auto &&base0, auto &&base1) {
        auto frame0 = make_frame(m0);
        auto frame1 = make_frame(m1);
        return compute_boolean(base0 | tf::tag(frame0), base1 | tf::tag(frame1), op);
      });
}

// With curves

auto make_boolean(polydata *input0, polydata *input1, tf::boolean_op op,
                  tf::return_curves_t)
    -> std::tuple<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>,
                  vtkSmartPointer<polydata>> {
  if (!input0 || !input1) {
    return {nullptr, nullptr, nullptr};
  }

  return dispatch(input0, input1, [op](auto &&base0, auto &&base1) {
    return compute_boolean_with_curves(base0, base1, op);
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

auto make_boolean(polydata *input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op, tf::return_curves_t)
    -> std::tuple<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>,
                  vtkSmartPointer<polydata>> {
  if (!input0 || !input1.first || !input1.second) {
    return {nullptr, nullptr, nullptr};
  }

  return dispatch(
      input0, input1.first, [op, m1 = input1.second](auto &&base0, auto &&base1) {
        auto frame1 = make_frame(m1);
        return compute_boolean_with_curves(base0, base1 | tf::tag(frame1), op);
      });
}

auto make_boolean(std::pair<polydata *, vtkMatrix4x4 *> input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op, tf::return_curves_t)
    -> std::tuple<vtkSmartPointer<polydata>, vtkSmartPointer<vtkSignedCharArray>,
                  vtkSmartPointer<polydata>> {
  if (!input0.first || !input1.first || !input0.second || !input1.second) {
    return {nullptr, nullptr, nullptr};
  }

  return dispatch(
      input0.first, input1.first,
      [op, m0 = input0.second, m1 = input1.second](auto &&base0, auto &&base1) {
        auto frame0 = make_frame(m0);
        auto frame1 = make_frame(m1);
        return compute_boolean_with_curves(base0 | tf::tag(frame0),
                                           base1 | tf::tag(frame1), op);
      });
}

} // namespace tf::vtk

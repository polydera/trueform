/**
 * @file tagged_operand.hpp
 * @brief A mesh tagged the way an operand reaches an arrangement build
 *
 * The tree, the face membership, the manifold edge link and the frame are
 * what a form publishes to the pipeline. A suite that builds one of those
 * structures by hand exercises tag completion instead of the code under
 * test, so every suite takes its operands from here.
 *
 * The transformation is always tagged — identity when the caller has none
 * — so one (index, real, arity) combination has exactly ONE form type, and
 * the compiled builders can name it.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include <trueform/core/frame.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/spatial/tree_config.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/manifold_edge_link.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

#include <oneapi/tbb/parallel_invoke.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace tf::test {

/// Owns the mesh and every structure tagged onto it, so `form()` stays valid
/// for the lifetime of the fixture.
template <typename Index, typename Real, std::size_t Ngon = 3>
struct tagged_operand {
  using buffer_t = tf::polygons_buffer<Index, Real, 3, Ngon>;

  buffer_t mesh;
  tf::aabb_tree<Index, Real, 3> tree;
  tf::face_membership<Index> membership;
  tf::manifold_edge_link<Index, Ngon> manifold;
  tf::transformation<Real, 3> placement;

  explicit tagged_operand(buffer_t input)
      : tagged_operand(std::move(input),
                       tf::make_identity_transformation<Real, 3>()) {}

  /// The structures are completed the way the pipeline completes them:
  /// the tree beside the membership and the edge link, not after them.
  tagged_operand(buffer_t input, tf::transformation<Real, 3> transformation)
      : mesh(std::move(input)), placement(std::move(transformation)) {
    tbb::parallel_invoke(
        // this-> keeps MSVC's delayed lambda parse off the tf::tree type.
        [&] { this->tree.build(mesh.polygons(), tf::config_tree(4, 12)); },
        [&] {
          membership.build(mesh.polygons());
          manifold.build(mesh.polygons().faces(), membership);
        });
  }

  auto form() {
    return mesh.polygons() | tf::tag(tree) | tf::tag(membership) |
           tf::tag(manifold) | tf::tag(placement);
  }
};

/// The operand of a mesh, its combination read off the mesh's own type.
template <typename Index, typename Real, std::size_t Ngon>
auto make_tagged_operand(tf::polygons_buffer<Index, Real, 3, Ngon> mesh)
    -> tagged_operand<Index, Real, Ngon> {
  return tagged_operand<Index, Real, Ngon>(std::move(mesh));
}

/// @overload The operand of a mesh that carries a placement.
template <typename Index, typename Real, std::size_t Ngon>
auto make_tagged_operand(tf::polygons_buffer<Index, Real, 3, Ngon> mesh,
                         tf::transformation<Real, 3> placement)
    -> tagged_operand<Index, Real, Ngon> {
  return tagged_operand<Index, Real, Ngon>(std::move(mesh),
                                           std::move(placement));
}

/// THE form type of a combination: what `tagged_operand::form()` returns.
template <typename Index, typename Real, std::size_t Ngon>
using form_t =
    decltype(std::declval<tagged_operand<Index, Real, Ngon> &>().form());

/// The range an N-operand build views. The forms behind it must outlive
/// the graph.
template <typename Index, typename Real, std::size_t Ngon>
using forms_range_t = tf::range<form_t<Index, Real, Ngon> *, tf::dynamic_size>;

/// The range an N-operand build views over the assembled forms. A vector's
/// own iterator is a distinct type from the pointer the builders are
/// instantiated on, so the range is always taken over the data pointer.
template <typename Form>
auto forms_range(std::vector<Form> &forms)
    -> tf::range<Form *, tf::dynamic_size> {
  return tf::make_range(forms.data(), forms.data() + forms.size());
}

/// The operands of an N-operand entry, assembled once into the vector the
/// graph's range views.
template <typename Index, typename Real, std::size_t Ngon>
auto tagged_forms(std::vector<tagged_operand<Index, Real, Ngon>> &operands)
    -> std::vector<form_t<Index, Real, Ngon>> {
  std::vector<form_t<Index, Real, Ngon>> forms;
  forms.reserve(operands.size());
  for (auto &operand : operands)
    forms.push_back(operand.form());
  return forms;
}

} // namespace tf::test

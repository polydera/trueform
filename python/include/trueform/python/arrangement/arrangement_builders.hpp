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
#pragma once
// The binding's compiled build tier. Every arrangement entry shares one
// prefix — the graph build — so the graph type is NAMED here structurally
// and BUILT in one translation unit per type combination. Every other
// binding TU declares the builder and compiles only its read tier.
//
// A builder is always declared with an explicit structural return type: a
// deduced return would instantiate the factory body in every consumer,
// which is exactly the cost this tier exists to move.
#include "../spatial/mesh.hpp"
#include <trueform/core/none.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/arrangement_graph.hpp>
#include <trueform/arrangement/policy/arrangement_pair_policy.hpp>
#include <trueform/arrangement/policy/arrangement_range_policy.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace tf::py {

/// The transformation an operand carries: the wrapper's when it has one,
/// identity otherwise. A form's TYPE never states the answer — the frame
/// is always tagged — so one type combination has exactly one form shape.
template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto form_transformation(const mesh_wrapper<Index, RealT, Ngon, Dims> &wrapper)
    -> tf::transformation<RealT, Dims> {
  if (wrapper.has_transformation())
    return tf::transformation<RealT, Dims>(wrapper.transformation_view());
  return tf::make_identity_transformation<RealT, Dims>();
}

/// THE tagged form of the arrangement tier: one tag order, every
/// structure present, the transformation stated as a value. Every binding
/// entry that builds an arrangement assembles its operands here and
/// nowhere else.
template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto tagged_form(mesh_wrapper<Index, RealT, Ngon, Dims> &wrapper,
                 const tf::transformation<RealT, Dims> &transformation) {
  return wrapper.make_primitive_range() | tf::tag(wrapper.tree()) |
         tf::tag(wrapper.face_membership()) |
         tf::tag(wrapper.manifold_edge_link()) | tf::tag(transformation);
}

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
using form_t = decltype(tagged_form(
    std::declval<mesh_wrapper<Index, RealT, Ngon, Dims> &>(),
    std::declval<const tf::transformation<RealT, Dims> &>()));

/// The operands of a list entry, assembled once into the vector the
/// graph's range views. The vector must outlive the graph.
template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto tagged_forms(
    std::vector<mesh_wrapper<Index, RealT, Ngon, Dims>> &wrappers)
    -> std::vector<form_t<Index, RealT, Ngon, Dims>> {
  std::vector<form_t<Index, RealT, Ngon, Dims>> forms;
  forms.reserve(wrappers.size());
  for (auto &wrapper : wrappers)
    forms.push_back(tagged_form(wrapper, form_transformation(wrapper)));
  return forms;
}

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
using forms_range_t =
    tf::range<form_t<Index, RealT, Ngon, Dims> *, tf::dynamic_size>;

// A tagged form carries tree, face membership and manifold edge link, so
// the dispatch layer builds nothing and the graph's structs type is
// tf::none_t. A builder TU proves it: its definition returns whatever
// tf::make_arrangement_graph deduced, against the type declared here.
template <typename Form>
using self_arrangement_t = tf::arrangement_graph<
    tf::arrangement::arrangement_range_policy<std::array<Form, 1>, tf::none_t>,
    tf::none_t>;

template <typename Form0, typename Form1>
using pair_arrangement_t =
    tf::arrangement_graph<tf::arrangement::arrangement_pair_policy<
                              Form0, tf::none_t, Form1, tf::none_t>,
                          tf::none_t>;

template <typename Forms>
using range_arrangement_t = tf::arrangement_graph<
    tf::arrangement::arrangement_range_policy<Forms, tf::none_t>, tf::none_t>;

/// The form's self arrangement (`within` implied).
template <typename Form>
auto build_self_arrangement(const Form &form, tf::arrangement_config config)
    -> self_arrangement_t<Form>;

/// Two operands, possibly of different types.
template <typename Form0, typename Form1>
auto build_pair_arrangement(const Form0 &form0, const Form1 &form1,
                            tf::arrangement_config config)
    -> pair_arrangement_t<Form0, Form1>;

/// N operands. The graph stores the range, so the forms behind it must
/// outlive the graph.
template <typename Forms>
auto build_range_arrangement(Forms forms, tf::arrangement_config config)
    -> range_arrangement_t<Forms>;

} // namespace tf::py

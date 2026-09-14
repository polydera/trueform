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
// The facade's compiled build tier. Every arrangement entry of every module
// shares one prefix -- the graph build -- so a graph type is NAMED here
// structurally and BUILT in one translation unit per operand-type combination.
// Every other shard declares the builder it needs and compiles only its read
// tier, which is what the core already publishes beside the fused entries
// (`tf::arrangement::arrangement_worker`, `tf::intersect::curves_worker`,
// `tf::make_csg_mesh`, `tf::make_outer_shell`).
//
// A builder is always declared with an explicit structural return type: a
// deduced return would instantiate the factory body in every consumer, which
// is exactly the cost this tier exists to move.

#include "trueform/arrangement/arrangement_config.hpp"
#include "trueform/arrangement/arrangement_graph.hpp"
#include "trueform/arrangement/policy/arrangement_pair_policy.hpp"
#include "trueform/arrangement/policy/arrangement_range_policy.hpp"
#include "trueform/core/none.hpp"
#include "trueform/core/range.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/csg/csg_graph.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace tf::cpp::graph_builders {

/// The form the arrangement tier reads a mesh through. Every shard assembles
/// its operands with `mesh::topology_form()` and nowhere else, so one
/// (index, real, arity) has exactly one form type and one build.
template <typename Index, typename Real, std::size_t Ngon>
using form_t = decltype(std::declval<const cpp::mesh<Index, Real, 3, Ngon> &>()
                            .topology_form());

/// The operands of a range entry as the graph views them. The forms behind it
/// must outlive the graph, and the range is spelled over their storage rather
/// than over a container's iterator, so one operand list is one type.
template <typename Index, typename Real, std::size_t Ngon>
using forms_range_t = tf::range<form_t<Index, Real, Ngon> *, tf::dynamic_size>;

/// The one sheet-list type the builders are compiled on.
using sheets_t = tf::range<const std::int32_t *, tf::dynamic_size>;

inline auto no_sheets() -> sheets_t {
  const std::int32_t *none = nullptr;
  return tf::make_range(none, none);
}

inline auto sheets_of(const std::vector<std::int32_t> &ids) -> sheets_t {
  const std::int32_t *first = ids.data();
  return tf::make_range(first, first + ids.size());
}

template <typename Forms>
auto forms_range(Forms &forms)
    -> tf::range<typename Forms::value_type *, tf::dynamic_size> {
  return tf::make_range(forms.data(), forms.data() + forms.size());
}

// A form carries tree, face membership and manifold edge link, so the
// dispatch layer builds nothing and the graph's structs type is tf::none_t.
// A builder TU proves it: its definition returns whatever the factory
// deduced, against the type declared here.
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

template <typename Form>
using self_csg_graph_t = tf::csg_graph<
    tf::arrangement::arrangement_range_policy<std::array<Form, 1>, tf::none_t>,
    tf::none_t, tf::arrangement_graph>;

template <typename Form0, typename Form1>
using pair_csg_graph_t =
    tf::csg_graph<tf::arrangement::arrangement_pair_policy<Form0, tf::none_t,
                                                           Form1, tf::none_t>,
                  tf::none_t, tf::arrangement_graph>;

template <typename Forms>
using range_csg_graph_t =
    tf::csg_graph<tf::arrangement::arrangement_range_policy<Forms, tf::none_t>,
                  tf::none_t, tf::arrangement_graph>;

/// The form's self arrangement (`within` implied).
template <typename Form>
auto build_self_arrangement(const Form &form, tf::arrangement_config config)
    -> self_arrangement_t<Form>;

/// Two operands, possibly of different index type or arity.
template <typename Form0, typename Form1>
auto build_pair_arrangement(const Form0 &form0, const Form1 &form1,
                            tf::arrangement_config config)
    -> pair_arrangement_t<Form0, Form1>;

/// N operands.
template <typename Forms>
auto build_range_arrangement(Forms forms, tf::arrangement_config config)
    -> range_arrangement_t<Forms>;

/// The same three, classified. A csg builder calls the arrangement builder
/// rather than re-instantiating the build, so a combination's arrangement is
/// compiled exactly once no matter how many tiers read it.
template <typename Form>
auto build_self_csg_graph(const Form &form, tf::arrangement_config config)
    -> self_csg_graph_t<Form>;

template <typename Form0, typename Form1>
auto build_pair_csg_graph(const Form0 &form0, const Form1 &form1,
                          sheets_t sheets, tf::arrangement_config config)
    -> pair_csg_graph_t<Form0, Form1>;

template <typename Forms>
auto build_range_csg_graph(Forms forms, sheets_t sheets,
                           tf::arrangement_config config)
    -> range_csg_graph_t<Forms>;

} // namespace tf::cpp::graph_builders

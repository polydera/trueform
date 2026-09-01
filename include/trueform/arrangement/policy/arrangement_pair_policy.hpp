/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/concatenated_blocked_range_collections.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/none.hpp"
#include "../../core/policy/none.hpp"
#include "../../core/range.hpp"
#include "../../core/static_size.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../exact/input_lattice.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/intersect_config.hpp"
#include "../../spatial/policy/tree.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "../construct/arrangement_map_data.hpp"
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::arrangement {

/// @brief Storage + construction policy of @ref tf::arrangement_graph
///        for TWO independently typed forms — the reason the pair
///        entry points exist. Heterogeneity is erased by a two-way
///        compile-time branch under the runtime tag inside
///        `apply_to_form(tag, f)`.
template <typename F0, typename S0, typename F1, typename S1>
struct arrangement_pair_policy {
  using index_type =
      std::common_type_t<std::decay_t<decltype(std::declval<F0>()
                                                   .faces()[0][0])>,
                         std::decay_t<decltype(std::declval<F1>()
                                                   .faces()[0][0])>>;
  using input_real_type = tf::coordinate_type<F0, F1>;

  /// Static face arity of the pair: triangles only when BOTH operands
  /// are triangles — a mixed pair must materialise per-face vertex
  /// counts, so it reports `tf::dynamic_size`.
  static constexpr std::size_t face_static_size =
      (tf::static_size_v<std::decay_t<decltype(std::declval<F0>()
                                                   .faces()[0])>> == 3 &&
       tf::static_size_v<std::decay_t<decltype(std::declval<F1>()
                                                   .faces()[0])>> == 3)
          ? std::size_t(3)
          : tf::dynamic_size;

  /// Two operands, always.
  static constexpr std::size_t static_n_tags = 2;

  arrangement_pair_policy(F0 form0, S0 structs0, F1 form1, S1 structs1)
      : _form0(std::move(form0)), _structs0(std::move(structs0)),
        _form1(std::move(form1)), _structs1(std::move(structs1)) {}

  auto n_tags() const -> index_type { return index_type(2); }

  auto tagged0() const {
    if constexpr (std::is_same_v<S0, tf::none_t>) {
      return _form0;
    } else {
      return std::apply(
          [&](const auto &...structs) {
            return (_form0 | ... | tf::tag(structs));
          },
          _structs0);
    }
  }
  auto tagged1() const {
    if constexpr (std::is_same_v<S1, tf::none_t>) {
      return _form1;
    } else {
      return std::apply(
          [&](const auto &...structs) {
            return (_form1 | ... | tf::tag(structs));
          },
          _structs1);
    }
  }

  auto make_apply_to_form() const {
    return [t0 = tagged0(), t1 = tagged1()](index_type tag, auto &&f) {
      if (tag == index_type(0))
        f(t0);
      else
        f(t1);
    };
  }

  /// @brief The operands' lattice view: one converter over their union,
  ///        the flat vertex space, and the door's placement when a
  ///        tolerance is given. The arity the converter is built at is
  ///        this policy's, exactly as `build_intersections`' is.
  template <typename Int> auto make_lattice(double tolerance) const {
    tf::exact::input_lattice<index_type, input_real_type, Int> lattice;
    lattice.build(tf::exact::make_vertex_converter<Int, input_real_type>(
                      tagged0(), tagged1()),
                  make_apply_to_form(), n_tags(), tolerance);
    return lattice;
  }

  template <typename Ibp, typename Lattice>
  auto build_intersections(Ibp &ibp, const Lattice &lattice,
                           tf::intersect_config config) const {
    ibp.build(tagged0(), tagged1(), lattice, config);
  }

  /// @brief The arrangement's output faces: each operand's uncut faces,
  ///        then the cut triangles.
  ///
  /// The two operands have different types, so their uncut views cannot
  /// share a range and are handed to the concatenation separately. That
  /// is the whole of what this policy does differently — the
  /// concatenation materialises, so the result is an owning buffer of
  /// the same type the homogeneous policy produces.
  template <typename Index, typename Triangles>
  auto concatenated_output_faces(
      const tf::arrangement::arrangement_map_data<Index> &map_data,
      const Triangles &triangles) const {
    auto original_maps = tf::make_offset_block_range(map_data.point_offsets,
                                                     map_data.original_map);
    auto uncut_of = [&](const auto &form, index_type t) {
      return tf::make_indirect_range(
          tf::make_range(map_data.original_face_ids[std::size_t(t)]),
          tf::make_block_indirect_range(
              form.faces(),
              tf::make_mapped_range(
                  original_maps[t],
                  [off = map_data.original_offsets[t]](Index x) {
                    return x + off;
                  })));
    };
    auto uncut0 = uncut_of(tagged0(), index_type(0));
    auto uncut1 = uncut_of(tagged1(), index_type(1));
    return tf::concatenated_blocked_range_collections<Index>(
        tf::make_range(&uncut0, 1), tf::make_range(&uncut1, 1),
        tf::make_range(&triangles, 1));
  }

private:
  F0 _form0;
  S0 _structs0;
  F1 _form1;
  S1 _structs1;
};

} // namespace tf::arrangement

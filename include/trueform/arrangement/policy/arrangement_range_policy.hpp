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
#include "../../core/small_vector.hpp"
#include "../../core/static_size.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/input_lattice.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/intersect_config.hpp"
#include "../../spatial/policy/tree.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "../construct/arrangement_map_data.hpp"
#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::arrangement {

/// @brief Storage + construction policy of @ref tf::arrangement_graph for a
///        homogeneous forms RANGE: the user's range plus the
///        owned structures the dispatch layer computed for missing
///        tags. Form access is `apply_to_form(tag, f)` — here a plain
///        index; the pair policy branches instead.
template <typename Forms, typename Structs> struct arrangement_range_policy {
  using index_type =
      std::decay_t<decltype(std::declval<Forms>()[0].faces()[0][0])>;
  using input_real_type =
      tf::coordinate_type<decltype(std::declval<Forms>()[0])>;

  /// Static face arity of the operands — `3` for a triangle mesh,
  /// `tf::dynamic_size` otherwise. The policy is the one producer:
  /// extraction picks its output buffer shape from it instead of
  /// re-deriving it from a forms element, which a heterogeneous policy
  /// has no single one of. Tagging never changes arity.
  static constexpr std::size_t face_static_size =
      tf::static_size_v<std::decay_t<decltype(std::declval<Forms>()[0]
                                                  .faces()[0])>>;

  /// Operand count when it is known at compile time — `1` for a single
  /// form (which arrives as a one-element array), `tf::dynamic_size` for
  /// a runtime range. Extraction reads it to choose what a result even
  /// means: one operand has no tag axis worth carrying.
  static constexpr std::size_t static_n_tags = tf::static_size_v<Forms>;

  arrangement_range_policy(Forms forms, tf::small_vector<Structs, 10> structs)
      : _forms(std::move(forms)), _structs(std::move(structs)) {}

  /// @brief The tagged forms view. If `Structs == none_t`, the user's
  ///        forms unchanged; otherwise a
  ///        `mapped_range(zip(_forms, _structs))` that tags every form
  ///        with the structures built for it.
  auto forms() const {
    if constexpr (std::is_same_v<Structs, tf::none_t>) {
      return _forms;
    } else {
      return tf::make_mapped_range(tf::zip(_forms, _structs), [](auto pair) {
        auto &&[form, s] = pair;
        return std::apply(
            [&form = form](const auto &...structs) {
              return (form | ... | tf::tag(structs));
            },
            s);
      });
    }
  }

  auto n_tags() const -> index_type {
    return static_cast<index_type>(forms().size());
  }

  /// @brief The two-phase form access every graph consumer programs
  ///        against: `apply_to_form(tag, f)` calls `f` with the
  ///        concrete tagged form.
  auto make_apply_to_form() const {
    return [tagged = forms()](index_type tag, auto &&f) { f(tagged[tag]); };
  }

  /// @brief The operands' lattice view: one converter over their union,
  ///        the flat vertex space, and the door's placement when a
  ///        tolerance is given. The arity the converter is built at is
  ///        this policy's, exactly as `build_intersections`' is.
  template <typename Int> auto make_lattice(double tolerance) const {
    // `forms()` is a value and `as_form_range` over a pre-tagged
    // operand array points INTO it, so the range is worth exactly as
    // much as the object it was taken from.
    const auto tagged = forms();
    tf::exact::input_lattice<index_type, input_real_type, Int> lattice;
    lattice.build(tf::exact::make_vertex_converter<Int, input_real_type>(
                      as_form_range(tagged)),
                  make_apply_to_form(), n_tags(), tolerance);
    return lattice;
  }

  template <typename Ibp, typename Lattice>
  auto build_intersections(Ibp &ibp, const Lattice &lattice,
                           tf::intersect_config config) const {
    ibp.build(as_form_range(forms()), lattice, config);
  }

  /// @brief The arrangement's output faces: every tag's uncut faces,
  ///        grouped by tag, then the cut triangles.
  ///
  /// The operands are one homogeneous range, so their uncut views share
  /// a type and ride one mapped range over the tags. This is the only
  /// step of the emission that depends on how the operands are stored,
  /// which is why the policy answers it; the concatenation materialises,
  /// so the result is an owning buffer whose type follows the arity
  /// alone.
  template <typename Index, typename Triangles>
  auto concatenated_output_faces(
      const tf::arrangement::arrangement_map_data<Index> &map_data,
      const Triangles &triangles) const {
    auto original_maps = tf::make_offset_block_range(map_data.point_offsets,
                                                     map_data.original_map);
    auto uncut = tf::make_mapped_range(
        tf::make_sequence_range(index_type(map_data.n_meshes)),
        [tagged = forms(), &map_data, original_maps](index_type t) {
          auto off = map_data.original_offsets[t];
          return tf::make_indirect_range(
              tf::make_range(map_data.original_face_ids[std::size_t(t)]),
              tf::make_block_indirect_range(
                  tagged[t].faces(),
                  tf::make_mapped_range(original_maps[t],
                                        [off](Index x) { return x + off; })));
        });
    return tf::concatenated_blocked_range_collections<Index>(
        uncut, tf::make_range(&triangles, 1));
  }

private:
  // pre-tagged single forms arrive as std::array; the intersections
  // build takes ranges
  template <typename T, std::size_t N>
  static auto as_form_range(const std::array<T, N> &forms) {
    return tf::make_range(forms.data(), forms.data() + N);
  }
  template <typename R> static auto as_form_range(const R &forms) -> R {
    return forms;
  }

  Forms _forms;
  tf::small_vector<Structs, 10> _structs;
};


} // namespace tf::arrangement

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
#include "../../core/coordinate_type.hpp"
#include "../../core/none.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/tagged_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/intersect_config.hpp"
#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::cut {

/// @brief Storage + construction policy of @ref tf::arrangement_graph for a
///        homogeneous forms RANGE: the user's range plus the
///        owned structures the dispatch layer computed for missing
///        tags. Form access is `apply_to_form(tag, f)` — here a plain
///        index; the pair policy branches instead.
template <typename Forms, typename Structs> struct arrangement_range_policy {
  using forms_type = Forms;
  using structs_type = Structs;
  using index_type =
      std::decay_t<decltype(std::declval<Forms>()[0].faces()[0][0])>;
  using input_real_type =
      tf::coordinate_type<decltype(std::declval<Forms>()[0])>;

  arrangement_range_policy(Forms forms, tf::small_vector<Structs, 10> structs)
      : _forms(std::move(forms)), _structs(std::move(structs)) {}

  /// @brief The tagged forms view. If `Structs == none_t`, the user's
  ///        forms unchanged; otherwise the same
  ///        `mapped_range(zip(_forms, _structs))` shape that
  ///        @ref tf::cut::dispatch::arrangement produces.
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

  template <typename Ibp>
  auto build_intersections(Ibp &ibp, tf::intersect_config config) const {
    ibp.build(as_form_range(forms()), config);
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


} // namespace tf::cut

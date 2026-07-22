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
#include "../../intersect/intersect_config.hpp"
#include "../../core/views/tagged_range.hpp"
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::cut {

/// @brief Storage + construction policy of @ref tf::arrangement_graph
///        for TWO independently typed forms — the reason the pair
///        entry points exist. Heterogeneity is erased exactly the way
///        the legacy exact pipeline always erased it: a two-way
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

  template <typename Ibp>
  auto build_intersections(Ibp &ibp, tf::intersect_config config) const {
    ibp.build(tagged0(), tagged1(), config);
  }

private:
  F0 _form0;
  S0 _structs0;
  F1 _form1;
  S1 _structs1;
};

} // namespace tf::cut

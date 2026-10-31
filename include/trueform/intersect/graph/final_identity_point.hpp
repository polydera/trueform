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

#include "../../core/config/verify.hpp"
#include "./plane_identity_collapse.hpp"

#include <cstdint>

namespace tf::intersect::graph {

/// An identity whose statement is being evaluated: the chain of them,
/// innermost first, encloses every read of that evaluation.
template <typename Index> struct evaluated_identity {
  Index id;
  bool through_merge;
  const evaluated_identity *enclosing;
};

/// THE ONE POSITION LAW, run on one layer's chase. A key speaks what
/// `target(source, id)` states for it (@ref plane_targeted_endpoint) —
/// except a key retired into an identity whose statement encloses the read,
/// which speaks its pre-mutation source at every depth of that evaluation.
/// `state(tag, id, self)` then reads what the key speaks, `self` enclosing
/// every read its statement makes. `retires` is the layer's own fact that it
/// retires at least one key; a layer retiring none answers every key with
/// itself, so it states `false` and no key is chased.
///
/// A read therefore either takes an endpoint its statement named, born
/// before the identity naming it, or enters a survivor no enclosing
/// statement holds, which grows the bounded enclosing set: every evaluation
/// is finite. A layer's tier evaluates at its own layer, so one identity has
/// one answer per layer and no second law.
template <typename Point, typename Index, typename Target,
          typename VertexOffsets, typename State>
auto final_identity_point(bool retires, const Target &target,
                          const VertexOffsets &vertex_offsets, std::int16_t tag,
                          Index id, const evaluated_identity<Index> *mints,
                          const State &state) -> Point {
  bool through_merge = false;
  std::int16_t merged_tag = tag;
  Index merged_id = id;
  if (retires)
    plane_targeted_endpoint(target, vertex_offsets, tag, id, merged_tag,
                            merged_id);
  if (merged_tag != tag || merged_id != id) {
    auto enclosing = mints;
    while (enclosing != nullptr && enclosing->id != merged_id)
      enclosing = enclosing->enclosing;
    if (enclosing == nullptr) {
      tag = merged_tag;
      id = merged_id;
      through_merge = true;
    }
  }
  // a survivor entered through the table holds no enclosing statement, and a
  // run of stated endpoints descends in birth order
  if constexpr (tf::core::verification)
    if (tag < 0)
      for (auto enclosing = mints; enclosing != nullptr;
           enclosing = !through_merge && enclosing->through_merge
                           ? nullptr
                           : enclosing->enclosing)
        tf::core::verify(enclosing->id != id,
                         "a name's evaluation revisits its own identity");
  const evaluated_identity<Index> self{id, through_merge, mints};
  return state(tag, id, &self);
}

} // namespace tf::intersect::graph

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
#include "../../core/views/sequence_range.hpp"

namespace tf::arrangement {

/// THE RENUMBERING LAW. A plane value numbers three positional spaces —
/// created identities, planes and faces — as `[world extent | promoted
/// suffix]`, each extent frozen when the value initializes. A diff's world
/// appends at its own end of every space, so the whole suffix slides by the
/// append count and nothing below the extent moves: between a prior value and
/// the next there is exactly this one renumbering, and it is monotone, so
/// among prior ids the numeric order the rank law and the close read as birth
/// order survives it. A gap id numbers below every forwarded suffix id by
/// construction: a round's own statements outrank the standing world at
/// their coincidences, so the round's stated survivor wins its ties.
/// Every reference a value carries into the next crosses it here and nowhere
/// else; a table survives a round only inside its owner, and the owner
/// crosses it through these.
///
/// `forward` takes a prior id into the next value and `inverse` the other
/// way, `-1` for an id in the `gap` — what the world appended, which no prior
/// holds. `prior_end` bounds the prior's own numbering, so an id outside the
/// numbering a read expects is refused there.
template <typename Index> struct extent_shift {
  Index prior;
  Index base;
  Index prior_end;

  auto forward(Index id) const -> Index {
    if constexpr (tf::core::verification)
      tf::core::verify(id < prior_end, "a forwarded id speaks the prior value");
    return id < prior ? id : id + (base - prior);
  }
  auto inverse(Index id) const -> Index {
    if constexpr (tf::core::verification)
      tf::core::verify(id < base + (prior_end - prior),
                       "an inverted id speaks the next value");
    return id < prior ? id : (id < base ? Index(-1) : id - (base - prior));
  }
  auto gap() const { return tf::make_sequence_range(prior, base); }
  /// The same shift in flat currency, past `n_flat` originals.
  auto lifted(Index n_flat) const -> extent_shift {
    return {n_flat + prior, n_flat + base, n_flat + prior_end};
  }
};

template <typename Index> struct diff_extents {
  extent_shift<Index> created;
  extent_shift<Index> planes;
  extent_shift<Index> faces;
};

/// The one producer of a diff's shifts: `prior`'s frozen extents, with the
/// `n_prior_created` identities its plane layer states past them, against
/// the extents of `world`, the restated world before any promotion is laid
/// over it.
template <typename Prior, typename World>
auto make_diff_extents(const Prior &prior,
                       typename World::index_type n_prior_created,
                       const World &world)
    -> diff_extents<typename World::index_type> {
  using Index = typename World::index_type;
  return {
      {prior.n_base_created(), world.n_created_points(),
       Index(prior.n_base_created() + n_prior_created)},
      {prior.n_base_planes(), world.n_planes(),
       Index(prior.n_base_planes() + Index(prior.promoted_frames().size()))},
      {prior.n_base_faces(), world.n_faces(),
       Index(prior.n_base_faces() +
             Index(prior.promoted_descriptors().size()))}};
}

} // namespace tf::arrangement

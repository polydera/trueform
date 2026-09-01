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

#include <type_traits>
#include <utility>

namespace tf::arrangement {

template <typename, typename, typename = std::void_t<>>
struct has_flat_carriers : std::false_type {};

template <typename World, typename Index>
struct has_flat_carriers<
    World, Index,
    std::void_t<decltype(std::declval<const World &>().carriers_of_flat(
        std::declval<Index>()))>> : std::true_type {};

/// THE CARRIER QUESTION — `carriers_of_flat(flat)`: which of a world's own
/// carriers can hold a row that names one flat identity, answered as a range
/// of planes. It is the identity's own MEMBERSHIP, which the mesh that owns
/// the identity already holds; a world answers by lookup, never by a pass
/// over its carriers.
///
/// THE ANSWER IS WHOLE OR THE NAME IS NOT STATED. A world states it only when
/// both hold of the rows of its own tier:
///
/// - an original vertex is named only as a CORNER of the face that emitted
///   the row, so the faces holding it are the whole carrier set;
/// - no row of that tier names a created identity, so a created identity is
///   the wave's alone and lives in the blocks the wave holds.
///
/// The second clause is why a created flat answers EMPTY rather than absent:
/// a consumer adds the blocks the wave holds, and the two together are every
/// carrier a retired identity can reach.
///
/// A world whose carrier space IS the answer — one whose every carrier was
/// cut, so every one of them can name a created identity — states nothing,
/// and its consumers sweep it whole. That is not a missing answer: the sweep
/// is then proportional to the arrangement itself, which is the set the
/// question would have returned.
template <typename World, typename Index>
inline constexpr bool states_flat_carriers =
    has_flat_carriers<std::decay_t<World>, Index>::value;

} // namespace tf::arrangement

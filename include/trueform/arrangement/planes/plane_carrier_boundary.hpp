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
struct has_carrier_boundary : std::false_type {};

template <typename World, typename Index>
struct has_carrier_boundary<
    World, Index,
    std::void_t<decltype(std::declval<const World &>().carrier_boundary(
        std::declval<Index>()))>> : std::true_type {};

/// THE BOUNDARY QUESTION — `carrier_boundary(plane)`: the corners of the face
/// a carrier IS, in the face's own loop order.
///
/// A WORLD STATES IT ONLY IF ITS CARRIER IS THAT FACE AND NOTHING ELSE — one
/// member, whose own boundary is the whole constraint set until a wave says
/// otherwise. That is what licenses a world to defer its definition tier: the
/// tier exists to carry a split between carriers, and a carrier that has taken
/// no split still reads exactly its own loop.
///
/// A world whose carriers are cut faces of an arrangement states nothing here:
/// its carrier's constraint set is the block its tier holds, which is the only
/// place the chords live.
template <typename World, typename Index>
inline constexpr bool states_carrier_boundary =
    has_carrier_boundary<std::decay_t<World>, Index>::value;

} // namespace tf::arrangement

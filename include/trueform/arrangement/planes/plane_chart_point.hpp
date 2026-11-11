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

template <typename, typename, typename, typename = std::void_t<>>
struct has_chart_point : std::false_type {};

template <typename World, typename Index, typename Point>
struct has_chart_point<
    World, Index, Point,
    std::void_t<decltype(std::declval<const World &>().chart_point(
        std::declval<Index>(), std::declval<Index>(),
        std::declval<const Point &>()))>> : std::true_type {};

/// THE CHART QUESTION — `chart_point(plane, flat, point)`: where a plane's
/// chart reads a flat identity whose one position is `point`. An identity
/// is one position for every tier that names it — the closure, the landing
/// law, the welds and a class's position all read that one — and only a
/// plane's own chart, its projected constraint set and its resolve, may read
/// it elsewhere, where the world says a point the plane states on its own
/// sides stands for that plane.
///
/// A world states it only if its planes chart a shared identity apart; a
/// world that states nothing charts every identity at its one position.
template <typename World, typename Index, typename Point>
inline constexpr bool states_chart_point =
    has_chart_point<std::decay_t<World>, Index, Point>::value;

} // namespace tf::arrangement

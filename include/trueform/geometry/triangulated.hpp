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
#include "../arrangement/mesh/mesh_triangulation.hpp"
#include "../clean/polygons.hpp"
#include "../core/is_soup.hpp"
#include "../core/none.hpp"
#include "../core/polygon.hpp"
#include "../core/polygons.hpp"
#include "../core/return_refused.hpp"
#include "./triangulation/make_loop_polygons.hpp"
#include "./triangulation/make_refused_faces.hpp"
#include "./triangulation/make_triangulated_mesh.hpp"
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup geometry_processing
/// @brief Triangulate all polygons and return a triangle mesh buffer.
///
/// The points are the input's own, then the identities the triangulation
/// minted; the triangles name them directly, so there is no remap. A face is
/// triangulated on its own boundary as the constraint set and a shared mesh
/// edge is one identity in both faces, so faces meeting on an edge stay
/// watertight.
///
/// THE FILL RULE IS `labels != 0`: a face's product is everything its
/// triangulation kept inside its own boundary, and the hull exterior is the
/// only thing dropped. For a simple loop that is the even-odd rule exactly;
/// for a loop that overlaps itself it is not — a doubly wound region is
/// inside, and this keeps it.
///
/// A FACE THAT NEEDS RESOLUTION IS RESOLVED, NOT DROPPED. A loop that crosses
/// itself states its crossing and mints the identity that names it, and the
/// mesh carries that identity in its own point table past the input's. THE
/// RESOLVED FACE'S POSITION IS UNCHANGED, ONLY ITS NAME: the election that
/// closes an identity ranks an original vertex last, so an original coincident
/// with another identity is retired into a minted one standing at exactly its
/// own coordinate.
///
/// AN ORIGINAL POINT GOES THROUGH THE FORM'S FRAME and a minted one through
/// the lattice this build quantized to, which puts both sides of the table in
/// ONE space: a minted point cannot be expressed in the untransformed space
/// without inverting the frame. On an untagged form the frame is the identity
/// and the points are the input's, copied.
///
/// A mesh no face of which needs resolving mints nothing, so the point table
/// is the input's and the triangles name only its own vertex ids.
///
/// A SOUP IS CLEANED TO SHARED-VERTEX IDENTITY FIRST — an exact-duplicate
/// clean makes a corner two faces state twice one point — so the triangulation
/// machinery sees only indexed meshes, and a shared edge is one identity there
/// as it is anywhere else.
///
/// @tparam Index The index type the mesh's faces are written in. It answers
/// two questions: the width, when an input carries an index type wider than
/// the caller needs — an `int64_t` mesh whose output fits `int32_t` asks for
/// `int32_t` and gets it; and the NAME, when the input is a soup and carries
/// no index type at all, in which case this is the caller's naming of the
/// output and defaults to `int`. For an indexed mesh it defaults to the
/// input's own.
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input polygons, indexed or a soup.
/// @return A polygons_buffer containing triangulated mesh (3 indices per face).
template <typename Index = tf::none_t, typename Policy>
auto triangulated(const tf::polygons<Policy> &polygons) {
  if constexpr (std::is_same_v<Index, tf::none_t> && tf::is_soup<Policy>) {
    // a soup carries no index type to default to, so the default is a fixed
    // one and the request is resolved by naming it
    return triangulated<int>(polygons);
  } else if constexpr (tf::is_soup<Policy>) {
    // the clean is what mints the shared-vertex identity a triangulation needs
    // to carry a split from a face into its neighbour; below it, nothing knows
    // a soup was ever here
    const auto indexed = tf::cleaned<Index>(polygons);
    return triangulated<Index>(indexed.polygons());
  } else {
    using OutIndex = std::conditional_t<
        std::is_same_v<Index, tf::none_t>,
        std::decay_t<decltype(polygons.faces()[0][0])>, Index>;
    return tf::geometry::make_triangulated_mesh<OutIndex>(
        tf::arrangement::make_mesh_triangulation(polygons), polygons);
  }
}

/// @ingroup geometry_processing
/// @brief Triangulate all polygons and name the faces the triangulator
///        refused.
///
/// The mesh is exactly the one the untagged call returns. A face whose
/// triangulation refused every round of the resolution wave holds no product
/// and contributes no triangle — emptiness is the answer, and this overload
/// names whose emptiness it was. A face the wave resolves is not refused, and
/// a face that bounds no area holds its product by emitting nothing, on the
/// same terms as every other carrier.
///
/// The ids are ascending face ids of the INPUT, a different identity space
/// from the mesh's corners, so they carry the input's own index type whatever
/// width the mesh was asked for. A soup has no such space to name — the clean
/// that gives it shared vertices drops the faces that bound nothing — so this
/// overload takes indexed meshes only.
///
/// @tparam Index The index type the mesh's faces are written in; defaults to
/// the input's own.
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input polygons.
/// @return Pair of (@ref tf::polygons_buffer, refused face ids).
template <typename Index = tf::none_t, typename Policy>
auto triangulated(const tf::polygons<Policy> &polygons, tf::return_refused_t) {
  static_assert(!tf::is_soup<Policy>,
                "A soup's faces do not survive the clean that gives it shared "
                "vertices, so it has no face identity to refuse.");
  using OutIndex =
      std::conditional_t<std::is_same_v<Index, tf::none_t>,
                         std::decay_t<decltype(polygons.faces()[0][0])>, Index>;
  const auto triangulation = tf::arrangement::make_mesh_triangulation(polygons);
  return std::make_pair(
      tf::geometry::make_triangulated_mesh<OutIndex>(triangulation, polygons),
      tf::geometry::make_refused_faces(triangulation));
}

/// @ingroup geometry_processing
/// @brief Triangulate a single polygon and return a triangle mesh buffer.
///
/// The loop is the one face of a one-face mesh, answered on the same terms as
/// every other face. A polygon that repeats its first point at the end draws
/// the same loop: the repeat is named by nothing and stays in the point table
/// where the caller put it.
///
/// The loop carries no ids of its own, so the mesh's faces are written in the
/// requested width directly.
///
/// @tparam Index The index type the mesh's faces are written in.
/// @tparam Dims The number of dimensions.
/// @tparam Policy The policy type of the polygon.
/// @param polygon The input polygon.
/// @return A polygons_buffer containing triangulated mesh (3 indices per face).
template <typename Index = tf::none_t, std::size_t Dims, typename Policy>
auto triangulated(const tf::polygon<Dims, Policy> &polygon) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    return triangulated<int>(polygon);
  } else {
    const auto loop = tf::geometry::make_loop_polygons<Index>(polygon);
    return tf::geometry::make_triangulated_mesh<Index>(
        tf::arrangement::make_mesh_triangulation(loop), loop);
  }
}

} // namespace tf

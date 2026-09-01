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

#include "../../core/buffer.hpp"
#include "../../core/coordinate_dims.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/none.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../core/transformed.hpp"
#include "../../exact/resolve_int_type.hpp"
#include "../../exact/vertex.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../planes/plane_arrangement.hpp"
#include "../planes/plane_arrangement_census.hpp"
#include "./plane_mesh_world.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::arrangement {

/// ONE TRIANGULATION PER FACE, on the tier that resolves. The mesh is the
/// plane arrangement's world, so a face's own boundary is its constraint set
/// and a shared mesh edge is one canonical group — which is what carries a
/// split from a face into its neighbour and keeps the two watertight.
///
/// A mesh that needs no resolution never enters one: no carrier refuses, so
/// the wave is one predicate, nothing is created and the output corners are
/// the input's own vertex ids.
///
/// The identity space is the input's: `vertex_offsets` is `{0, n_points}`, so
/// a corner below `n_original_points()` IS the mesh vertex id and a corner at
/// or past it indexes `created_points()`. Created points stay on the lattice
/// this build quantized to; `converter()` is what returns them to reals.
///
/// THE FILL RULE IS `labels != 0`: a face's product is everything its
/// triangulation kept inside its own boundary, and the hull exterior is the
/// only thing dropped. For a simple loop that is the even-odd rule exactly;
/// for a loop that overlaps itself it is not — a doubly wound region is
/// inside, and this keeps it.
///
/// A RESOLVED FACE NEED NOT NAME EVERY VERTEX IT WAS GIVEN. The election that
/// closes a wave's identities ranks an original vertex LAST and never elects
/// one as a component's survivor, so an original that coincides with another
/// identity — two corners quantizing to one lattice point, or a corner a
/// landing names — is retired into a minted identity standing at exactly its
/// own coordinate. The product's corner is then a created id where the input
/// had a vertex id. The position is unchanged; only the name is.
template <typename Index, typename RealT, typename Int, std::size_t Dims,
          typename Faces>
class mesh_triangulation {
public:
  using index_type = Index;
  using real_type = RealT;
  using int_type = Int;
  using point_type = tf::exact::pt3<Int>;
  using world_type = plane_mesh_world<Index, Int, Faces>;
  using converter_type = tf::exact::vertex_converter<Int, RealT, Dims>;

  /// The world is a VIEW of this mesh, so it is constructed here and the build
  /// below reads it — there is no empty triangulation to fill in later.
  template <typename Policy>
  explicit mesh_triangulation(const tf::polygons<Policy> &polygons)
      : _converter(tf::exact::make_vertex_converter<Int, RealT>(polygons)),
        _world(make_world(polygons, _converter)),
        _n_original_points(Index(polygons.points().size())) {
    static_assert(tf::coordinate_dims_v<Policy> == Dims,
                  "the polygons' dimension is the triangulation's");
    static_assert(
        std::is_same<std::decay_t<decltype(polygons.faces())>, Faces>::value,
        "the world carries the polygons' own faces");
    _vertex_offsets.allocate(2);
    _vertex_offsets[0] = Index(0);
    _vertex_offsets[1] = _n_original_points;
  }

  auto build() -> void {
    // ONE READER, and it is the world's own table: the conversion is stated
    // once per vertex, not once per corner that names it
    const auto read = [this](std::int16_t, Index id) -> point_type {
      return _world.point(id);
    };
    _arr.build(_world, Index(0), read, read, _vertex_offsets);

    // a created identity may name another, so the table is filled in the
    // order the tier minted them
    _created_points.allocate(std::size_t(_arr.n_created()));
    for (Index id = 0; id < _arr.n_created(); ++id)
      _created_points[std::size_t(id)] =
          _arr.resolve_created_point(id, read, read);
  }

  /// Every triangle, contiguous per input face. A corner is a flat identity:
  /// below `n_original_points()` the input's own vertex id, past it a row of
  /// `created_points()`.
  auto triangles() const { return _arr.triangles(); }
  /// Per triangle, per corner: where the corner sits on its face's own
  /// polygon — its corner ordinal, the original side it lies on, or the
  /// interior. The provenance an attribute transfer reads.
  auto corner_subs() const { return _arr.corner_subs(); }
  /// One input face's own triangle span.
  auto face_range(Index face) const -> std::array<Index, 2> {
    return _arr.face_range(face);
  }
  auto n_faces() const -> Index { return _arr.n_faces(); }
  auto n_original_points() const -> Index { return _n_original_points; }
  /// THE COMPLETENESS SURFACE, forwarded: a plane IS a face here, so these are
  /// the input faces whose triangulation refused every round of the wave.
  /// Empty means every face holds its product; a face that bounds no area
  /// holds its product by emitting nothing, on the same terms as every other
  /// carrier.
  auto failed() const { return _arr.failed(); }
  /// The identities this build minted, on the lattice, indexed by
  /// `corner - n_original_points()`.
  auto created_points() const { return tf::make_range(_created_points); }
  auto converter() const -> const converter_type & { return _converter; }
  auto world() const -> const world_type & { return _world; }
  auto census() const -> const plane_arrangement_census & {
    return _arr.census();
  }

private:
  /// The mesh on the lattice this build quantized to. The lattice is
  /// three-dimensional whatever the input is: a plane of a two-dimensional
  /// mesh is the lattice's own z = 0.
  template <typename Policy>
  static auto make_world(const tf::polygons<Policy> &polygons,
                         const converter_type &converter) -> world_type {
    const auto frame = tf::frame_of(polygons);
    return make_plane_mesh_world<Index, Int>(
        polygons.faces(), Index(polygons.points().size()),
        [&](Index id) -> point_type {
          const auto converted = converter.convert(
              tf::transformed(polygons.points()[std::size_t(id)], frame));
          if constexpr (Dims == 2) {
            point_type lifted;
            lifted[0] = converted[0];
            lifted[1] = converted[1];
            lifted[2] = Int(0);
            return lifted;
          } else {
            return converted;
          }
        });
  }

  converter_type _converter;
  world_type _world;
  plane_arrangement<Index, Int> _arr;
  tf::buffer<Index> _vertex_offsets;
  tf::buffer<point_type> _created_points;
  Index _n_original_points;
};

/// The triangulation of a mesh on the resolving tier. `Int` is the lattice the
/// exact predicates run on; it is resolved against the input's own real type.
template <typename Int = tf::none_t, typename Policy>
auto make_mesh_triangulation(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using RealT = tf::coordinate_type<Policy>;
  mesh_triangulation<Index, RealT, tf::exact::resolve_int_type<Int, RealT>,
                     tf::coordinate_dims_v<Policy>,
                     std::decay_t<decltype(polygons.faces())>>
      product(polygons);
  product.build();
  return product;
}

} // namespace tf::arrangement

#pragma once

#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/tetrahedra_buffer.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/exact/resolve_int_type.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/tetmesh/entrance/tet_entrance.hpp>
#include <trueform/tetmesh/tet_mesh_config.hpp>
#include <trueform/tetmesh/tet_mesh_index_map.hpp>
#include <trueform/tetmesh/tet_mesh_report.hpp>
#include <trueform/tetmesh/tet_mesh_status.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/manifold_edge_link.hpp>

#include <cstdint>
#include <tuple>

namespace tf::test {

using float_tet_entrance_t = tf::tetmesh::tet_entrance<
    int, float, tf::exact::resolve_int_type<tf::none_t, float>, std::int64_t>;

/// The tet entrance of a float triangle mesh, compiled once.
auto build_tet_entrance(const tf::points_buffer<float, 3> &points,
                        const tf::blocked_buffer<int, 3> &faces)
    -> float_tet_entrance_t;

using float_tet_mesh_t =
    std::tuple<tf::tetrahedra_buffer<int, float>, tf::tet_mesh_report,
               tf::tet_mesh_index_map<int>>;

/// tf::make_tet_mesh of a float triangle mesh with its index map, compiled
/// once.
auto build_tet_mesh(const tf::polygons_buffer<int, float, 3, 3> &mesh)
    -> float_tet_mesh_t;

/// tf::make_tet_mesh of a float triangle mesh on the int64 lattice, the
/// deeper one a caller states where the int32 lattice leaves crossings
/// unresolved, compiled once.
auto build_tet_mesh_on_int64(const tf::polygons_buffer<int, float, 3, 3> &mesh)
    -> float_tet_mesh_t;

using double_tet_mesh_t =
    std::tuple<tf::tetrahedra_buffer<int, double>, tf::tet_mesh_report,
               tf::tet_mesh_index_map<int>>;

using default_tet_mesh_t =
    std::tuple<tf::tetrahedra_buffer<int, double>, tf::tet_mesh_status>;

/// tf::make_tet_mesh's default call on a float triangle mesh, compiled once:
/// the declared type is what the call returns.
auto build_tet_mesh_default(const tf::polygons_buffer<int, float, 3, 3> &mesh)
    -> default_tet_mesh_t;

/// tf::make_tet_mesh of a float triangle mesh emitted in double under
/// `config`, compiled once: `{false}` takes the surface as given.
auto build_tet_mesh_double(const tf::polygons_buffer<int, float, 3, 3> &mesh,
                           const tf::tet_mesh_config &config = {})
    -> double_tet_mesh_t;

/// tf::make_tet_mesh of a float triangle mesh tagged with `transformation`,
/// emitted in double, compiled once.
auto build_tet_mesh_framed(const tf::polygons_buffer<int, float, 3, 3> &mesh,
                           const tf::transformation<float, 3> &transformation)
    -> double_tet_mesh_t;

/// tf::make_tet_mesh of a float triangle mesh tagged with `membership` and
/// `link`, emitted in double, compiled once.
auto build_tet_mesh_tagged(const tf::polygons_buffer<int, float, 3, 3> &mesh,
                           const tf::face_membership<int> &membership,
                           const tf::manifold_edge_link<int, 3> &link)
    -> double_tet_mesh_t;

/// tf::make_tet_mesh of a float triangle mesh tagged with `tree` alone,
/// emitted in double, compiled once.
auto build_tet_mesh_with_tree(const tf::polygons_buffer<int, float, 3, 3> &mesh,
                              const tf::aabb_tree<int, float, 3> &tree)
    -> double_tet_mesh_t;

/// tf::make_tet_mesh of a float triangle mesh tagged with its manifold edge
/// link alone, emitted in double, compiled once.
auto build_tet_mesh_with_link(const tf::polygons_buffer<int, float, 3, 3> &mesh,
                              const tf::manifold_edge_link<int, 3> &link)
    -> double_tet_mesh_t;

/// tf::make_tet_mesh of a double triangle mesh, compiled once.
auto build_tet_mesh(const tf::polygons_buffer<int, double, 3, 3> &mesh)
    -> double_tet_mesh_t;

} // namespace tf::test

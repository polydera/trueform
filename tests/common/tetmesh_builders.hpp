#pragma once

#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/tetrahedra_buffer.hpp>
#include <trueform/exact/resolve_int_type.hpp>
#include <trueform/tetmesh/entrance/tet_entrance.hpp>
#include <trueform/tetmesh/tet_mesh_report.hpp>

#include <array>
#include <cstdint>
#include <tuple>

namespace tf::test {

using float_tet_entrance_t =
    tf::tetmesh::tet_entrance<int, float,
                              tf::exact::resolve_int_type<tf::none_t, float>,
                              std::int64_t>;

/// The tet entrance of a float triangle mesh, compiled once.
auto build_tet_entrance(const tf::points_buffer<float, 3> &points,
                        const tf::blocked_buffer<int, 3> &faces)
    -> float_tet_entrance_t;

using float_tet_mesh_t =
    std::tuple<tf::tetrahedra_buffer<int, float>,
               tf::offset_block_buffer<int, std::array<int, 3>>,
               tf::offset_block_buffer<int, int>, tf::tet_mesh_report>;

/// tf::make_tet_mesh of a float triangle mesh, compiled once.
auto build_tet_mesh(const tf::polygons_buffer<int, float, 3, 3> &mesh)
    -> float_tet_mesh_t;

using double_tet_mesh_t =
    std::tuple<tf::tetrahedra_buffer<int, double>,
               tf::offset_block_buffer<int, std::array<int, 3>>,
               tf::offset_block_buffer<int, int>, tf::tet_mesh_report>;

/// tf::make_tet_mesh of a float triangle mesh emitted in double, compiled
/// once.
auto build_tet_mesh_double(const tf::polygons_buffer<int, float, 3, 3> &mesh)
    -> double_tet_mesh_t;

/// tf::make_tet_mesh of a double triangle mesh, compiled once.
auto build_tet_mesh(const tf::polygons_buffer<int, double, 3, 3> &mesh)
    -> double_tet_mesh_t;

} // namespace tf::test

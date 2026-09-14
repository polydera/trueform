#include "axis_shard_test_utils.hpp"

#include <trueform/cpp/topology/cell_membership.hpp>
#include <trueform/cpp/topology/face_link.hpp>
#include <trueform/cpp/topology/manifold_edge_link.hpp>
#include <trueform/cpp/topology/vertex_link.hpp>

#include <cstdint>
#include <iostream>

namespace {

template <typename Index>
auto equals(const tf::cpp::nd_array<Index> &actual,
            std::initializer_list<Index> expected) -> bool {
  return actual.length() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

template <typename Index>
auto equals(const tf::cpp::offset_blocked_buffer<Index, Index> &actual,
            std::initializer_list<Index> offsets,
            std::initializer_list<Index> data) -> bool {
  return equals(actual.offsets(), offsets) && equals(actual.data(), data);
}

} // namespace

int main() {
  using index_type = std::int64_t;
  using namespace trueform_installed_test;
  const auto fixed_faces = make_array<index_type>({0, 1, 2, 1, 3, 2}, {2, 3});
  const auto dynamic_faces =
      tf::cpp::offset_blocked_buffer<index_type, index_type>::create(
          make_array<index_type>({0, 3, 6}, {3}),
          make_array<index_type>({0, 1, 2, 1, 3, 2}, {6}));
  const auto edges = make_array<index_type>({0, 1, 1, 2, 2, 3}, {3, 2});
  const auto edge_membership = tf::cpp::cell_membership(edges, index_type{4});
  const auto fixed_membership =
      tf::cpp::cell_membership(fixed_faces, index_type{4});
  const auto dynamic_membership =
      tf::cpp::cell_membership(dynamic_faces, index_type{4});
  const auto edge_vertex_link =
      tf::cpp::vertex_link_edges(edges, index_type{4});
  const auto fixed_vertex_link =
      tf::cpp::vertex_link_faces(fixed_faces, fixed_membership);
  const auto dynamic_vertex_link =
      tf::cpp::vertex_link_faces(dynamic_faces, dynamic_membership);
  const auto fixed_manifold =
      tf::cpp::manifold_edge_link(fixed_faces, fixed_membership);
  const auto dynamic_manifold =
      tf::cpp::manifold_edge_link(dynamic_faces, dynamic_membership);
  const auto fixed_face = tf::cpp::face_link(fixed_faces, fixed_membership);
  const auto dynamic_face =
      tf::cpp::face_link(dynamic_faces, dynamic_membership);

  const auto ok =
      equals(edge_membership,
             {index_type{0}, index_type{1}, index_type{3}, index_type{5},
              index_type{6}},
             {index_type{0}, index_type{1}, index_type{0}, index_type{2},
              index_type{1}, index_type{2}}) &&
      equals(fixed_membership,
             {index_type{0}, index_type{1}, index_type{3}, index_type{5},
              index_type{6}},
             {index_type{0}, index_type{1}, index_type{0}, index_type{1},
              index_type{0}, index_type{1}}) &&
      equals(dynamic_membership,
             {index_type{0}, index_type{1}, index_type{3}, index_type{5},
              index_type{6}},
             {index_type{0}, index_type{1}, index_type{0}, index_type{1},
              index_type{0}, index_type{1}}) &&
      equals(edge_vertex_link,
             {index_type{0}, index_type{1}, index_type{3}, index_type{5},
              index_type{6}},
             {index_type{1}, index_type{2}, index_type{0}, index_type{3},
              index_type{1}, index_type{2}}) &&
      equals(fixed_vertex_link,
             {index_type{0}, index_type{2}, index_type{5}, index_type{8},
              index_type{10}},
             {index_type{2}, index_type{1}, index_type{2}, index_type{3},
              index_type{0}, index_type{3}, index_type{1}, index_type{0},
              index_type{1}, index_type{2}}) &&
      equals(dynamic_vertex_link,
             {index_type{0}, index_type{2}, index_type{5}, index_type{8},
              index_type{10}},
             {index_type{2}, index_type{1}, index_type{2}, index_type{3},
              index_type{0}, index_type{3}, index_type{1}, index_type{0},
              index_type{1}, index_type{2}}) &&
      equals(fixed_manifold, {index_type{-1}, index_type{1}, index_type{-1},
                              index_type{-1}, index_type{-1}, index_type{0}}) &&
      equals(dynamic_manifold, {index_type{0}, index_type{3}, index_type{6}},
             {index_type{-1}, index_type{1}, index_type{-1}, index_type{-1},
              index_type{-1}, index_type{0}}) &&
      equals(fixed_face, {index_type{0}, index_type{1}, index_type{2}},
             {index_type{1}, index_type{0}}) &&
      equals(dynamic_face, {index_type{0}, index_type{1}, index_type{2}},
             {index_type{1}, index_type{0}});
  if (!ok) {
    std::cerr << "topology int64 links mismatch\nfixed manifold:";
    for (const auto value : fixed_manifold)
      std::cerr << ' ' << value;
    std::cerr << "\ndynamic manifold offsets/data:";
    for (const auto value : dynamic_manifold.offsets())
      std::cerr << ' ' << value;
    std::cerr << " /";
    for (const auto value : dynamic_manifold.data())
      std::cerr << ' ' << value;
    std::cerr << "\nfixed face offsets/data:";
    for (const auto value : fixed_face.offsets())
      std::cerr << ' ' << value;
    std::cerr << " /";
    for (const auto value : fixed_face.data())
      std::cerr << ' ' << value;
    std::cerr << "\ndynamic face offsets/data:";
    for (const auto value : dynamic_face.offsets())
      std::cerr << ' ' << value;
    std::cerr << " /";
    for (const auto value : dynamic_face.data())
      std::cerr << ' ' << value;
    std::cerr << '\n';
  }
  return ok ? 0 : 1;
}

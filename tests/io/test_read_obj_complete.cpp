/**
 * @file test_read_obj_complete.cpp
 * @brief Tests for the complete-mode OBJ reader (positions, normals,
 *        textures, groups, objects) and writer banners.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <trueform/io.hpp>

namespace {

auto temp_path(const char *suffix) -> std::filesystem::path {
  static std::atomic<int> counter{0};
  static auto pid = std::random_device{}();
  auto id = counter.fetch_add(1);
  auto name = std::string("tf_complete_") + std::to_string(pid) + "_" +
              std::to_string(id) + suffix;
  return std::filesystem::temp_directory_path() / name;
}

struct cleanup {
  std::filesystem::path p;
  ~cleanup() { std::filesystem::remove(p); }
};

auto write_text(const std::filesystem::path &p, const std::string &s) -> void {
  std::ofstream f(p);
  f << s;
}

// Find the unique vertex id with the given (x, y, z) position.
auto find_pt(const tf::obj_file<int> &f, float x, float y, float z) -> int {
  for (std::size_t i = 0; i < f.polygons.points().size(); ++i) {
    auto p = f.polygons.points()[i];
    if (p[0] == x && p[1] == y && p[2] == z)
      return static_cast<int>(i);
  }
  return -1;
}

} // namespace

// =============================================================================
// Dedup correctness
// =============================================================================

TEST_CASE("dedup: identical (v,vt,vn) triplets across faces collapse to one",
          "[io][read_obj][complete][dedup]") {
  // Quad written as two triangles, sharing v=1 and v=3 with identical UVs.
  // Expect 4 unique vertices (one per quad corner), not 6 (corner count).
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                "f 1/1 2/2 3/3\n"
                "f 1/1 3/3 4/4\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 4);
  REQUIRE(f.textures.size() == 4);
  REQUIRE(f.polygons.faces().size() == 2);
}

TEST_CASE("dedup: position values come back unchanged",
          "[io][read_obj][complete][dedup]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 2 0 0\nv 0 3 0\n"
                "f 1 2 3\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 3);
  // Find each input position in the output (ID may be permuted by sort).
  REQUIRE(find_pt(f, 0, 0, 0) >= 0);
  REQUIRE(find_pt(f, 2, 0, 0) >= 0);
  REQUIRE(find_pt(f, 0, 3, 0) >= 0);
}

TEST_CASE("dedup: face indices reference correct positions",
          "[io][read_obj][complete][dedup]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 5 0 0\nv 0 7 0\n"
                "f 1 2 3\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.faces().size() == 1);
  auto face = f.polygons.faces()[0];
  REQUIRE(face.size() == 3);
  // Collect the (x,y,z) of the three corners.
  std::set<std::tuple<float, float, float>> got;
  for (auto vi : face) {
    auto pt = f.polygons.points()[vi];
    got.insert({pt[0], pt[1], pt[2]});
  }
  REQUIRE(got.count({0, 0, 0}) == 1);
  REQUIRE(got.count({5, 0, 0}) == 1);
  REQUIRE(got.count({0, 7, 0}) == 1);
}

TEST_CASE("dedup: texture seam - same position, different vt, splits vertex",
          "[io][read_obj][complete][dedup]") {
  // v=1 used twice with different UVs: (vt=1) then (vt=2).
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0.0 0.0\nvt 0.5 0.5\n"
                "f 1/1 2/1 3/1\n"
                "f 1/2 2/1 3/1\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  // 4 unique triplets: (1,1)(2,1)(3,1)(1,2). v=1 appears twice as a position.
  REQUIRE(f.polygons.points().size() == 4);
  REQUIRE(f.textures.size() == 4);

  // Count how many output vertices are at position (0,0,0): should be 2
  // (split copies of v=1).
  int n_at_origin = 0;
  for (std::size_t i = 0; i < f.polygons.points().size(); ++i) {
    auto pt = f.polygons.points()[i];
    if (pt[0] == 0 && pt[1] == 0 && pt[2] == 0)
      ++n_at_origin;
  }
  REQUIRE(n_at_origin == 2);

  // The two copies must carry different texture coords: one has (0,0), the
  // other (0.5, 0.5).
  std::set<std::pair<float, float>> uvs_at_origin;
  for (std::size_t i = 0; i < f.polygons.points().size(); ++i) {
    auto pt = f.polygons.points()[i];
    if (pt[0] == 0 && pt[1] == 0 && pt[2] == 0) {
      auto uv = f.textures[i];
      uvs_at_origin.insert({uv[0], uv[1]});
    }
  }
  REQUIRE(uvs_at_origin.size() == 2);
  REQUIRE(uvs_at_origin.count({0.0f, 0.0f}) == 1);
  REQUIRE(uvs_at_origin.count({0.5f, 0.5f}) == 1);
}

TEST_CASE("dedup: sharp edge - same position, different vn, splits vertex",
          "[io][read_obj][complete][dedup]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vn 0 0 1\nvn 1 0 0\n"
                "f 1//1 2//1 3//1\n"
                "f 1//2 2//1 3//1\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 4);
  REQUIRE(f.normals.size() == 4);
  REQUIRE(f.textures.size() == 0);

  std::set<std::tuple<float, float, float>> normals_at_origin;
  for (std::size_t i = 0; i < f.polygons.points().size(); ++i) {
    auto pt = f.polygons.points()[i];
    if (pt[0] == 0 && pt[1] == 0 && pt[2] == 0) {
      auto n = f.normals[i];
      normals_at_origin.insert({n[0], n[1], n[2]});
    }
  }
  REQUIRE(normals_at_origin.size() == 2);
  REQUIRE(normals_at_origin.count({0.0f, 0.0f, 1.0f}) == 1);
  REQUIRE(normals_at_origin.count({1.0f, 0.0f, 0.0f}) == 1);
}

TEST_CASE("dedup: aligned attribute values are correct",
          "[io][read_obj][complete][dedup]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "vt 0.1 0.2\nvt 0.3 0.4\nvt 0.5 0.6\n"
                "vn 1 0 0\nvn 0 1 0\nvn 0 0 1\n"
                "f 1/1/1 2/2/2 3/3/3\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(f.textures.size() == 3);
  REQUIRE(f.normals.size() == 3);
  // For each output vertex i, normals[i] and textures[i] correspond to the
  // SAME triplet that produced points[i]. Verify by checking the bijection.
  for (std::size_t i = 0; i < f.polygons.points().size(); ++i) {
    auto pt = f.polygons.points()[i];
    auto n = f.normals[i];
    auto uv = f.textures[i];
    if (pt[0] == 1 && pt[1] == 0 && pt[2] == 0) {
      REQUIRE(n[0] == 1.0f);
      REQUIRE(uv[0] == 0.1f);
      REQUIRE(uv[1] == 0.2f);
    } else if (pt[0] == 0 && pt[1] == 1 && pt[2] == 0) {
      REQUIRE(n[1] == 1.0f);
      REQUIRE(uv[0] == 0.3f);
      REQUIRE(uv[1] == 0.4f);
    } else if (pt[0] == 0 && pt[1] == 0 && pt[2] == 1) {
      REQUIRE(n[2] == 1.0f);
      REQUIRE(uv[0] == 0.5f);
      REQUIRE(uv[1] == 0.6f);
    } else {
      FAIL("unexpected position");
    }
  }
}

// =============================================================================
// N-gon faces
// =============================================================================

TEST_CASE("complete: quad face", "[io][read_obj][complete][ngon]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                "f 1 2 3 4\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.faces().size() == 1);
  REQUIRE(f.polygons.faces()[0].size() == 4);
}

TEST_CASE("complete: pentagon face", "[io][read_obj][complete][ngon]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 2 0 0\nv 3 1 0\nv 1 2 0\nv -1 1 0\n"
                "f 1 2 3 4 5\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.faces().size() == 1);
  REQUIRE(f.polygons.faces()[0].size() == 5);
}

// =============================================================================
// Groups & objects
// =============================================================================

TEST_CASE("groups: contiguous block per group, label sequence",
          "[io][read_obj][complete][groups]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nv 2 0 0\nv 2 1 0\n"
                "g A\n"
                "f 1 2 3\n"
                "f 2 4 3\n"
                "f 1 2 4\n"
                "g B\n"
                "f 2 5 4\n"
                "f 5 6 4\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.faces().size() == 5);
  REQUIRE(f.group_names.size() == 2);
  REQUIRE(f.group_names[0] == "A");
  REQUIRE(f.group_names[1] == "B");
  REQUIRE(f.face_groups.size() == 5);
  REQUIRE(f.face_groups[0] == 0);
  REQUIRE(f.face_groups[1] == 0);
  REQUIRE(f.face_groups[2] == 0);
  REQUIRE(f.face_groups[3] == 1);
  REQUIRE(f.face_groups[4] == 1);
}

TEST_CASE("groups: faces before first g get default group at index 0",
          "[io][read_obj][complete][groups]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
                "f 1 2 3\n" // implicit default
                "g Top\n"
                "f 1 2 4\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.group_names.size() == 2);
  REQUIRE(f.group_names[0] == "default");
  REQUIRE(f.group_names[1] == "Top");
  REQUIRE(f.face_groups[0] == 0);
  REQUIRE(f.face_groups[1] == 1);
}

TEST_CASE("groups: g-only file populates groups, leaves objects empty",
          "[io][read_obj][complete][groups]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
                "g Object_0\n"
                "f 1 2 3\n"
                "g Object_1\n"
                "f 2 4 3\n"
                "g Object_2\n"
                "f 1 4 3\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.group_names.size() == 3);
  REQUIRE(f.group_names[0] == "Object_0");
  REQUIRE(f.group_names[1] == "Object_1");
  REQUIRE(f.group_names[2] == "Object_2");
  REQUIRE(f.object_names.empty());
  REQUIRE(f.face_objects.size() == 0);
}

TEST_CASE("objects + groups: independent label sequences",
          "[io][read_obj][complete][groups]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
                "o Mesh\n"
                "g part_a\n"
                "f 1 2 3\n"
                "g part_b\n"
                "f 1 2 4\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.object_names.size() == 1);
  REQUIRE(f.object_names[0] == "Mesh");
  REQUIRE(f.face_objects.size() == 2);
  REQUIRE(f.face_objects[0] == 0);
  REQUIRE(f.face_objects[1] == 0);
  REQUIRE(f.group_names.size() == 2);
  REQUIRE(f.group_names[0] == "part_a");
  REQUIRE(f.group_names[1] == "part_b");
  REQUIRE(f.face_groups[0] == 0);
  REQUIRE(f.face_groups[1] == 1);
}

// =============================================================================
// Error / edge cases
// =============================================================================

TEST_CASE("error: mixed face refs (v vs v/vt/vn) returns empty obj_file",
          "[io][read_obj][complete][error]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0 0\nvt 1 0\nvt 0 1\n"
                "f 1 2 3\n"
                "f 1/1 2/2 3/3\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 0);
  REQUIRE(f.polygons.faces().size() == 0);
}

TEST_CASE("error: out-of-range vertex index returns empty",
          "[io][read_obj][complete][error]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\n"
                "f 1 2 3\n"); // only 2 vertices, face references 3
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 0);
  REQUIRE(f.polygons.faces().size() == 0);
}

TEST_CASE("complete: comments and unknown directives skipped",
          "[io][read_obj][complete]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "# trueform - https://trueform.polydera.com\n"
                "mtllib foo.mtl\n"
                "usemtl bar\n"
                "s off\n"
                "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "f 1 2 3\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(f.polygons.faces().size() == 1);
}

TEST_CASE("complete: relative indices currently unsupported",
          "[io][read_obj][complete]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 0);
}

TEST_CASE("complete: file with no faces returns empty",
          "[io][read_obj][complete]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.faces().size() == 0);
  REQUIRE(f.polygons.points().size() == 0);
}

TEST_CASE("complete: handles excessive whitespace and empty lines",
          "[io][read_obj][complete]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "\n\n  v   0 0 0  \n\n  v 1 0 0\n\nv 0 1 0\n\n  f  1  2  3  \n\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(f.polygons.faces().size() == 1);
}

TEST_CASE("complete: no duplication when attributes are consistent",
          "[io][read_obj][complete][dedup]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  // 3 points, 1 texture coord, 1 normal.
  // All faces share the same normal/texture at each point.
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0.5 0.5\n"
                "vn 0 0 1\n"
                "f 1/1/1 2/1/1 3/1/1\n"
                "f 1/1/1 3/1/1 2/1/1\n"); // Reverse face, same triplets
  auto f = tf::read_obj(p.string(), tf::complete);

  // Should be exactly 3 points, matching the 'v' count.
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(f.normals.size() == 3);
  REQUIRE(f.textures.size() == 3);

  // Verify values are correctly broadcasted
  for (std::size_t i = 0; i < 3; ++i) {
    auto n = f.normals[i];
    auto uv = f.textures[i];
    REQUIRE(n[2] == 1.0f);
    REQUIRE(uv[0] == 0.5f);
    REQUIRE(uv[1] == 0.5f);
  }
}

TEST_CASE("complete: comprehensive attribute verification",
          "[io][read_obj][complete][dedup]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  // 3 unique vertices with unique P, N, and T
  write_text(p, "v 1.1 0 0\nv 0 2.2 0\nv 0 0 3.3\n"
                "vt 0.1 0.1\nvt 0.2 0.2\nvt 0.3 0.3\n"
                "vn 1 0 0\nvn 0 1 0\nvn 0 0 1\n"
                "f 1/1/1 2/2/2 3/3/3\n");
  auto f = tf::read_obj(p.string(), tf::complete);

  REQUIRE(f.polygons.points().size() == 3);

  // Since the output order is sorted by (v, vt, vn) triplet,
  // we use find_pt to verify the exact cross-attribute integrity.
  int id1 = find_pt(f, 1.1f, 0.0f, 0.0f);
  int id2 = find_pt(f, 0.0f, 2.2f, 0.0f);
  int id3 = find_pt(f, 0.0f, 0.0f, 3.3f);

  REQUIRE(id1 >= 0);
  REQUIRE(f.textures[id1][0] == 0.1f);
  REQUIRE(f.normals[id1][0] == 1.0f);

  REQUIRE(id2 >= 0);
  REQUIRE(f.textures[id2][0] == 0.2f);
  REQUIRE(f.normals[id2][1] == 1.0f);

  REQUIRE(id3 >= 0);
  REQUIRE(f.textures[id3][0] == 0.3f);
  REQUIRE(f.normals[id3][2] == 1.0f);
}

// =============================================================================
// Writer banners
// =============================================================================

TEST_CASE("write_obj: first line is the trueform banner",
          "[io][write_obj][banner]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  tf::polygons_buffer<int, float, 3, 3> mesh;
  mesh.points_buffer().allocate(3);
  mesh.points()[0] = tf::point<float, 3>{0, 0, 0};
  mesh.points()[1] = tf::point<float, 3>{1, 0, 0};
  mesh.points()[2] = tf::point<float, 3>{0, 1, 0};
  mesh.faces_buffer().push_back({0, 1, 2});
  REQUIRE(tf::write_obj(mesh.polygons(), p.string()));

  std::ifstream f(p);
  std::string line;
  std::getline(f, line);
  auto expected = std::string("# ") + tf::io::trueform_banner;
  REQUIRE(line == expected);
}

TEST_CASE("write_stl: 80-byte header carries the trueform banner",
          "[io][write_stl][banner]") {
  auto p = temp_path(".stl");
  cleanup g{p};
  tf::polygons_buffer<int, float, 3, 3> mesh;
  mesh.points_buffer().allocate(3);
  mesh.points()[0] = tf::point<float, 3>{0, 0, 0};
  mesh.points()[1] = tf::point<float, 3>{1, 0, 0};
  mesh.points()[2] = tf::point<float, 3>{0, 1, 0};
  mesh.faces_buffer().push_back({0, 1, 2});
  REQUIRE(tf::write_stl(mesh.polygons(), p.string()));

  std::ifstream f(p, std::ios::binary);
  char header[80] = {};
  f.read(header, 80);
  auto banner_len = std::strlen(tf::io::trueform_banner);
  REQUIRE(std::strncmp(header, tf::io::trueform_banner, banner_len) == 0);
}

// =============================================================================
// API surface: Index template + memory-buffer overload
// =============================================================================

TEST_CASE("complete: int64_t Index template parameter",
          "[io][read_obj][complete][api]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
  auto f = tf::read_obj<int64_t>(p.string(), tf::complete);
  static_assert(std::is_same_v<decltype(f), tf::obj_file<int64_t>>);
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(f.polygons.faces().size() == 1);
}

TEST_CASE("complete: memory-buffer overload",
          "[io][read_obj][complete][api]") {
  std::string data = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
  auto rng = tf::make_range(data.data(), data.size());
  auto f = tf::read_obj(rng, tf::complete);
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(f.polygons.faces().size() == 1);
}

// =============================================================================
// Lexical / format edge cases
// =============================================================================

TEST_CASE("complete: mixed n-gon faces in same file",
          "[io][read_obj][complete][ngon]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\n"
                "f 1 2 3 4\n" // quad
                "f 2 5 3\n"); // triangle
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.faces().size() == 2);
  REQUIRE(f.polygons.faces()[0].size() == 4);
  REQUIRE(f.polygons.faces()[1].size() == 3);
}

TEST_CASE("complete: CRLF line endings", "[io][read_obj][complete]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nf 1 2 3\r\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(f.polygons.faces().size() == 1);
}

TEST_CASE("complete: inline trailing comment after face",
          "[io][read_obj][complete]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3 # trailing\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(f.polygons.faces().size() == 1);
  REQUIRE(f.polygons.faces()[0].size() == 3);
}

TEST_CASE("complete: negative vt index returns empty",
          "[io][read_obj][complete][error]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0 0\nvt 1 0\nvt 0 1\n"
                "f 1/-1 2/-1 3/-1\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 0);
}

TEST_CASE("complete: empty file returns empty",
          "[io][read_obj][complete][error]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 0);
  REQUIRE(f.polygons.faces().size() == 0);
}

TEST_CASE("round-trip: write then read complete works through banner",
          "[io][read_obj][complete][banner]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  tf::polygons_buffer<int, float, 3, 3> mesh;
  mesh.points_buffer().allocate(3);
  mesh.points()[0] = tf::point<float, 3>{0, 0, 0};
  mesh.points()[1] = tf::point<float, 3>{1, 0, 0};
  mesh.points()[2] = tf::point<float, 3>{0, 1, 0};
  mesh.faces_buffer().push_back({0, 1, 2});
  REQUIRE(tf::write_obj(mesh.polygons(), p.string()));

  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(f.polygons.faces().size() == 1);
  REQUIRE(find_pt(f, 0, 0, 0) >= 0);
  REQUIRE(find_pt(f, 1, 0, 0) >= 0);
  REQUIRE(find_pt(f, 0, 1, 0) >= 0);
}

// =============================================================================
// Line partitions: files large enough for the reader to split them
// =============================================================================

namespace {

// `obj_execution_tuning::target_partition_bytes` is 256 KiB, so a file of a
// few megabytes is read as many independent line partitions.
auto complete_position_block(int n) -> std::string {
  std::string s;
  for (int i = 0; i < n; ++i)
    s += "v " + std::to_string(i) + " " + std::to_string(i % 7) + " 0\n";
  return s;
}

// The tests below write faces over three consecutive positions, so corner k
// of face i must come back as the position the file wrote at 3i + k.
auto complete_faces_name_their_positions(const tf::obj_file<int> &f) -> bool {
  auto faces = f.polygons.faces();
  auto points = f.polygons.points();
  for (std::size_t i = 0; i < faces.size(); ++i) {
    auto face = faces[i];
    if (face.size() != 3)
      return false;
    for (std::size_t k = 0; k < 3; ++k) {
      auto expected = static_cast<float>(3 * i + k);
      auto point = points[face[k]];
      if (point[0] != expected || point[1] != std::fmod(expected, 7.0f))
        return false;
    }
  }
  return true;
}

} // namespace

TEST_CASE("partitions: groups keep their faces across line partitions",
          "[io][read_obj][complete][groups][partitions]") {
  const int n = 90000;
  const int per_group = 3750;
  auto p = temp_path(".obj");
  cleanup g{p};
  std::string s = complete_position_block(n);
  int face = 0;
  for (int i = 0; i + 2 < n; i += 3) {
    if (face % per_group == 0)
      s += "g grp_" + std::to_string(face / per_group) + "\n";
    ++face;
    s += "f " + std::to_string(i + 1) + " " + std::to_string(i + 2) + " " +
         std::to_string(i + 3) + "\n";
  }
  REQUIRE(s.size() > 4 * 256 * 1024);
  write_text(p, s);

  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.faces().size() == static_cast<std::size_t>(face));
  REQUIRE(f.group_names.size() == static_cast<std::size_t>(face / per_group));
  REQUIRE(f.group_names.front() == "grp_0");
  REQUIRE(f.group_names.back() ==
          "grp_" + std::to_string(face / per_group - 1));
  REQUIRE(f.face_groups.size() == static_cast<std::size_t>(face));
  bool labelled = true;
  for (int i = 0; i < face; ++i)
    labelled = labelled && f.face_groups[i] == i / per_group;
  REQUIRE(labelled);
  REQUIRE(complete_faces_name_their_positions(f));
}

TEST_CASE("partitions: one late group leaves every earlier face on default",
          "[io][read_obj][complete][groups][partitions]") {
  const int n = 90000;
  const int before = 10;
  auto p = temp_path(".obj");
  cleanup g{p};
  std::string s = complete_position_block(n);
  int face = 0;
  for (int i = 0; i + 2 < n; i += 3) {
    if (face == before)
      s += "g late\n";
    ++face;
    s += "f " + std::to_string(i + 1) + " " + std::to_string(i + 2) + " " +
         std::to_string(i + 3) + "\n";
  }
  REQUIRE(s.size() > 4 * 256 * 1024);
  write_text(p, s);

  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.group_names.size() == 2);
  REQUIRE(f.group_names[0] == "default");
  REQUIRE(f.group_names[1] == "late");
  bool labelled = true;
  for (int i = 0; i < face; ++i)
    labelled = labelled && f.face_groups[i] == (i < before ? 0 : 1);
  REQUIRE(labelled);
}

TEST_CASE("partitions: objects and groups carry independently",
          "[io][read_obj][complete][groups][partitions]") {
  const int n = 90000;
  const int per_object = 9000;
  const int per_group = 1500;
  auto p = temp_path(".obj");
  cleanup g{p};
  std::string s = complete_position_block(n);
  int face = 0;
  for (int i = 0; i + 2 < n; i += 3) {
    if (face % per_object == 0)
      s += "o obj_" + std::to_string(face / per_object) + "\n";
    if (face % per_group == 0)
      s += "g grp_" + std::to_string(face / per_group) + "\n";
    ++face;
    s += "f " + std::to_string(i + 1) + " " + std::to_string(i + 2) + " " +
         std::to_string(i + 3) + "\n";
  }
  write_text(p, s);

  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.object_names.size() ==
          static_cast<std::size_t>((face + per_object - 1) / per_object));
  REQUIRE(f.group_names.size() ==
          static_cast<std::size_t>((face + per_group - 1) / per_group));
  bool labelled = true;
  for (int i = 0; i < face; ++i)
    labelled = labelled && f.face_objects[i] == i / per_object &&
               f.face_groups[i] == i / per_group;
  REQUIRE(labelled);
}

TEST_CASE("partitions: attribute seams split vertices and stay aligned",
          "[io][read_obj][complete][dedup][partitions]") {
  // Position v is named with attribute 2v and 2v+1, so it becomes two output
  // vertices whose texture and normal must agree on which one they came from.
  const int n = 20000;
  auto p = temp_path(".obj");
  cleanup g{p};
  std::string s = complete_position_block(n);
  for (int a = 0; a < 2 * n; ++a)
    s += "vt " + std::to_string(a) + " 0\n";
  for (int a = 0; a < 2 * n; ++a)
    s += "vn " + std::to_string(a) + " 0 1\n";
  for (int i = 0; i + 2 < n; ++i) {
    s += "f";
    for (int k = 0; k < 3; ++k) {
      int v = i + k;
      int a = 1 + 2 * v + (i % 2);
      s += " " + std::to_string(v + 1) + "/" + std::to_string(a) + "/" +
           std::to_string(a);
    }
    s += "\n";
  }
  REQUIRE(s.size() > 4 * 256 * 1024);
  write_text(p, s);

  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() > static_cast<std::size_t>(n));
  REQUIRE(f.normals.size() == f.polygons.points().size());
  REQUIRE(f.textures.size() == f.polygons.points().size());
  bool aligned = true;
  for (std::size_t k = 0; k < f.polygons.points().size(); ++k) {
    auto attribute = static_cast<int>(f.textures[k][0]);
    aligned = aligned && f.normals[k][0] == f.textures[k][0] &&
              f.polygons.points()[k][0] == static_cast<float>(attribute / 2);
  }
  REQUIRE(aligned);
}

TEST_CASE("partitions: the face mode may be stated by a later partition",
          "[io][read_obj][complete][partitions]") {
  const int n = 60000;
  auto p = temp_path(".obj");
  cleanup g{p};
  std::string s = complete_position_block(n);
  for (int i = 0; i < n; ++i)
    s += "vn 0 0 1\n";
  for (int i = 0; i + 2 < n; i += 3) {
    s += "f";
    for (int k = 0; k < 3; ++k) {
      auto v = std::to_string(i + 1 + k);
      s += " " + v + "//" + v;
    }
    s += "\n";
  }
  REQUIRE(s.size() > 4 * 256 * 1024);
  write_text(p, s);

  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == static_cast<std::size_t>(n));
  REQUIRE(f.normals.size() == static_cast<std::size_t>(n));
  REQUIRE(f.textures.size() == 0);
  REQUIRE(complete_faces_name_their_positions(f));
}

TEST_CASE("partitions: a face mode change across partitions returns empty",
          "[io][read_obj][complete][error][partitions]") {
  const int n = 90000;
  auto p = temp_path(".obj");
  cleanup g{p};
  std::string s = complete_position_block(n);
  for (int i = 0; i < n; ++i)
    s += "vt 0.5 0.5\n";
  for (int i = 0; i + 2 < n; i += 3) {
    const bool textured = i > n / 2;
    s += "f";
    for (int k = 0; k < 3; ++k) {
      auto v = std::to_string(i + 1 + k);
      s += " " + v + (textured ? "/" + v : "");
    }
    s += "\n";
  }
  write_text(p, s);

  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 0);
  REQUIRE(f.polygons.faces().size() == 0);
}

TEST_CASE("partitions: mixed arity faces keep their spans",
          "[io][read_obj][complete][ngon][partitions]") {
  const int n = 90000;
  auto p = temp_path(".obj");
  cleanup g{p};
  std::string s = complete_position_block(n);
  int face = 0;
  for (int i = 0; i + 6 < n; i += 6) {
    int arity = 3 + face % 4;
    s += "f";
    for (int k = 0; k < arity; ++k)
      s += " " + std::to_string(i + 1 + k);
    s += "\n";
    ++face;
  }
  REQUIRE(s.size() > 4 * 256 * 1024);
  write_text(p, s);

  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.faces().size() == static_cast<std::size_t>(face));
  bool spanned = true;
  for (int i = 0; i < face; ++i)
    spanned = spanned && f.polygons.faces()[i].size() ==
                             static_cast<std::size_t>(3 + i % 4);
  REQUIRE(spanned);
}

TEST_CASE("complete: positions no face names are dropped",
          "[io][read_obj][complete][dedup]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 9 9 9\nv 1 0 0\nv 8 8 8\nv 0 1 0\nf 1 3 5\n");
  auto f = tf::read_obj(p.string(), tf::complete);
  REQUIRE(f.polygons.points().size() == 3);
  REQUIRE(find_pt(f, 9, 9, 9) == -1);
  REQUIRE(find_pt(f, 8, 8, 8) == -1);
  REQUIRE(find_pt(f, 0, 0, 0) >= 0);
  REQUIRE(find_pt(f, 1, 0, 0) >= 0);
  REQUIRE(find_pt(f, 0, 1, 0) >= 0);
}

TEST_CASE("complete: a face token the parser refuses returns empty",
          "[io][read_obj][complete][error]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf /1 2 3\n");
  REQUIRE(tf::read_obj(p.string(), tf::complete).polygons.points().size() == 0);
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1a/2 3 4\n");
  REQUIRE(tf::read_obj(p.string(), tf::complete).polygons.points().size() == 0);
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/\n");
  REQUIRE(tf::read_obj(p.string(), tf::complete).polygons.points().size() == 0);
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf # only a comment\n");
  REQUIRE(tf::read_obj(p.string(), tf::complete).polygons.points().size() == 0);
}

// =============================================================================
// Cross-reader agreement: the three entries on one file
// =============================================================================

namespace {

// Every position is named by a face, so the complete read's deduplicated
// vertices are the file's positions in file order and the three readers can be
// compared corner for corner.
auto cross_triangle_obj(int n_positions) -> std::string {
  std::string s;
  for (int i = 0; i < n_positions; ++i)
    s += "v " + std::to_string(i) + " " + std::to_string(i % 7) + " " +
         std::to_string(i % 3) + "\n";
  for (int i = 0; i + 2 < n_positions; i += 3)
    s += "f " + std::to_string(i + 1) + " " + std::to_string(i + 2) + " " +
         std::to_string(i + 3) + "\n";
  return s;
}

template <typename A, typename B>
auto cross_same_points(const A &left, const B &right) -> bool {
  if (left.size() != right.size())
    return false;
  for (std::size_t i = 0; i < left.size(); ++i)
    for (std::size_t k = 0; k < 3; ++k)
      if (left[i][k] != right[i][k])
        return false;
  return true;
}

} // namespace

TEST_CASE("cross: the positions-only and complete reads agree on geometry",
          "[io][read_obj][complete][cross]") {
  const int n = 90000;
  auto p = temp_path(".obj");
  cleanup g{p};
  auto s = cross_triangle_obj(n);
  REQUIRE(s.size() > 4 * 256 * 1024);
  write_text(p, s);

  auto dynamic_mesh = tf::read_obj(p.string());
  auto complete_mesh = tf::read_obj(p.string(), tf::complete);

  REQUIRE(cross_same_points(dynamic_mesh.points(),
                            complete_mesh.polygons.points()));
  REQUIRE(dynamic_mesh.faces().size() == complete_mesh.polygons.faces().size());
  bool same_faces = true;
  for (std::size_t i = 0; i < dynamic_mesh.faces().size(); ++i) {
    auto left = dynamic_mesh.faces()[i];
    auto right = complete_mesh.polygons.faces()[i];
    same_faces = same_faces && left.size() == right.size();
    if (!same_faces)
      break;
    for (std::size_t k = 0; k < left.size(); ++k)
      same_faces = same_faces && left[k] == right[k];
  }
  REQUIRE(same_faces);
}

TEST_CASE("cross: the fixed-arity and positions-only reads agree on triangles",
          "[io][read_obj][cross]") {
  const int n = 90000;
  auto p = temp_path(".obj");
  cleanup g{p};
  auto s = cross_triangle_obj(n);
  REQUIRE(s.size() > 4 * 256 * 1024);
  write_text(p, s);

  auto fixed_mesh = tf::read_obj<int, 3>(p.string());
  auto dynamic_mesh = tf::read_obj(p.string());

  REQUIRE(cross_same_points(fixed_mesh.points(), dynamic_mesh.points()));
  REQUIRE(fixed_mesh.faces().size() == dynamic_mesh.faces().size());
  bool same_faces = true;
  for (std::size_t i = 0; i < fixed_mesh.faces().size(); ++i) {
    auto left = fixed_mesh.faces()[i];
    auto right = dynamic_mesh.faces()[i];
    same_faces = same_faces && right.size() == 3;
    if (!same_faces)
      break;
    for (std::size_t k = 0; k < 3; ++k)
      same_faces = same_faces && left[k] == right[k];
  }
  REQUIRE(same_faces);
}

TEST_CASE("cross: a face with too few corners is dropped by one reader and "
          "refused by the others",
          "[io][read_obj][complete][cross][error]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
                "f 1 2 3\n"
                "f 1 2\n"
                "f 2 4 3\n");

  // The mixed-arity read drops the short face and keeps the rest.
  auto dynamic_mesh = tf::read_obj(p.string());
  REQUIRE(dynamic_mesh.faces().size() == 2);
  REQUIRE(dynamic_mesh.faces()[1][0] == 1);
  REQUIRE(dynamic_mesh.faces()[1][1] == 3);
  REQUIRE(dynamic_mesh.faces()[1][2] == 2);

  // The complete read refuses the file outright.
  auto complete_mesh = tf::read_obj(p.string(), tf::complete);
  REQUIRE(complete_mesh.polygons.faces().size() == 0);
  REQUIRE(complete_mesh.polygons.points().size() == 0);

  // The fixed-arity read refuses it too: the face is not an `Ngon`.
  auto fixed_mesh = tf::read_obj<int, 3>(p.string());
  REQUIRE(fixed_mesh.faces().size() == 0);
}

TEST_CASE("cross: an out-of-range vertex index is refused by all three "
          "readers",
          "[io][read_obj][complete][cross][error]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "f 1 2 3\n"
                "f 1 2 999999\n");

  auto dynamic_mesh = tf::read_obj(p.string());
  REQUIRE(dynamic_mesh.faces().size() == 0);
  REQUIRE(dynamic_mesh.points().size() == 0);

  auto fixed_mesh = tf::read_obj<int, 3>(p.string());
  REQUIRE(fixed_mesh.faces().size() == 0);
  REQUIRE(fixed_mesh.points().size() == 0);

  auto complete_mesh = tf::read_obj(p.string(), tf::complete);
  REQUIRE(complete_mesh.polygons.faces().size() == 0);
  REQUIRE(complete_mesh.polygons.points().size() == 0);
}

// =============================================================================
// Stated arity: read_obj<Ngon>(path, tf::complete)
// =============================================================================

namespace {

// An attributed all-triangle file with groups and objects, larger than the
// reader's partition target.
auto arity_attributed_obj(int n_positions) -> std::string {
  std::string s;
  for (int i = 0; i < n_positions; ++i)
    s += "v " + std::to_string(i) + " " + std::to_string(i % 7) + " 0\n";
  for (int i = 0; i < n_positions; ++i)
    s += "vt " + std::to_string(i % 977) + " 0\n";
  for (int i = 0; i < n_positions; ++i)
    s += "vn " + std::to_string(i % 13) + " 0 1\n";
  int face = 0;
  for (int i = 0; i + 2 < n_positions; i += 3) {
    if (face % 4000 == 0) {
      s += "o obj_" + std::to_string(face / 4000) + "\n";
      s += "g grp_" + std::to_string(face / 4000) + "\n";
    }
    ++face;
    s += "f";
    for (int k = 0; k < 3; ++k) {
      auto v = std::to_string(i + 1 + k);
      s += " " + v + "/" + v + "/" + v;
    }
    s += "\n";
  }
  return s;
}

} // namespace

TEST_CASE("arity: a stated arity reads what the mixed-size read reads",
          "[io][read_obj][complete][arity][partitions]") {
  const int n = 60000;
  auto p = temp_path(".obj");
  cleanup g{p};
  auto s = arity_attributed_obj(n);
  REQUIRE(s.size() > 4 * 256 * 1024);
  write_text(p, s);

  auto mixed = tf::read_obj(p.string(), tf::complete);
  auto fixed = tf::read_obj<3>(p.string(), tf::complete);
  static_assert(std::is_same_v<decltype(fixed), tf::obj_file<int, float, 3>>);

  REQUIRE(fixed.polygons.points().size() == mixed.polygons.points().size());
  REQUIRE(fixed.polygons.faces().size() == mixed.polygons.faces().size());
  REQUIRE(fixed.normals.size() == mixed.normals.size());
  REQUIRE(fixed.textures.size() == mixed.textures.size());
  REQUIRE(fixed.group_names == mixed.group_names);
  REQUIRE(fixed.object_names == mixed.object_names);

  bool same_vertices = true;
  for (std::size_t i = 0; i < mixed.polygons.points().size(); ++i) {
    for (std::size_t k = 0; k < 3; ++k)
      same_vertices = same_vertices && fixed.polygons.points()[i][k] ==
                                           mixed.polygons.points()[i][k] &&
                      fixed.normals[i][k] == mixed.normals[i][k];
    for (std::size_t k = 0; k < 2; ++k)
      same_vertices =
          same_vertices && fixed.textures[i][k] == mixed.textures[i][k];
  }
  REQUIRE(same_vertices);

  bool same_faces = true;
  for (std::size_t i = 0; i < mixed.polygons.faces().size(); ++i) {
    auto left = fixed.polygons.faces()[i];
    auto right = mixed.polygons.faces()[i];
    same_faces = same_faces && right.size() == 3 &&
                 fixed.face_groups[i] == mixed.face_groups[i] &&
                 fixed.face_objects[i] == mixed.face_objects[i];
    if (!same_faces)
      break;
    for (std::size_t k = 0; k < 3; ++k)
      same_faces = same_faces && left[k] == right[k];
  }
  REQUIRE(same_faces);
}

TEST_CASE("arity: a face of another size refuses the read",
          "[io][read_obj][complete][arity][error]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                "f 1 2 3\n"
                "f 1 2 3 4\n");
  auto fixed = tf::read_obj<3>(p.string(), tf::complete);
  REQUIRE(fixed.polygons.faces().size() == 0);
  REQUIRE(fixed.polygons.points().size() == 0);

  // The mixed-size read takes the same file.
  auto mixed = tf::read_obj(p.string(), tf::complete);
  REQUIRE(mixed.polygons.faces().size() == 2);
}

TEST_CASE("arity: a quad file read as quads",
          "[io][read_obj][complete][arity]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n");
  auto quads = tf::read_obj<4>(p.string(), tf::complete);
  REQUIRE(quads.polygons.faces().size() == 1);
  REQUIRE(quads.polygons.points().size() == 4);
}

TEST_CASE("arity: the Index-first and Ngon-first forms agree",
          "[io][read_obj][complete][arity][api]") {
  auto p = temp_path(".obj");
  cleanup g{p};
  write_text(p, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
  auto a = tf::read_obj<int64_t, 3>(p.string(), tf::complete);
  static_assert(std::is_same_v<decltype(a), tf::obj_file<int64_t, float, 3>>);
  auto b = tf::read_obj<3, int64_t>(p.string(), tf::complete);
  static_assert(std::is_same_v<decltype(b), tf::obj_file<int64_t, float, 3>>);
  REQUIRE(a.polygons.faces().size() == 1);
  REQUIRE(b.polygons.faces().size() == 1);

  std::string data = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
  auto c = tf::read_obj<3>(tf::make_range(data.data(), data.size()),
                           tf::complete);
  static_assert(std::is_same_v<decltype(c), tf::obj_file<int, float, 3>>);
  REQUIRE(c.polygons.faces().size() == 1);
}

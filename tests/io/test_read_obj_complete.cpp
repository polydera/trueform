/**
 * @file test_read_obj_complete.cpp
 * @brief Tests for the complete-mode OBJ reader (positions, normals,
 *        textures, groups, objects) and writer banners.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <atomic>
#include <catch2/catch_test_macros.hpp>
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

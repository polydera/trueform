/**
 * @file test_split_into_domains.cpp
 * @brief Tests for tf::split_into_domains(...)
 *
 * The domain labels are the instrument here; what is asserted is the
 * shape of what the split returns — one mesh per kept domain, source
 * blocks running parallel to the meshes, one source face id per emitted
 * face in emitted-face order, and an empty input coming back empty.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/trueform.hpp>

TEST_CASE("split_into_domains return_source_ids (box provenance)",
          "[reindex][domains][source_ids]") {
  using index_t = int;
  using real_t = float;

  auto box = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
  tf::orient_faces_consistently(box.polygons());

  auto labels = tf::make_domain_labels(box.polygons());
  // A closed box bounds two domains: the interior and the outer shell.
  REQUIRE(labels.n_domains == 2);

  auto [meshes, domain_ids, source] =
      tf::split_into_domains(box.polygons(), labels, tf::return_source_ids);

  // Source blocks run parallel to the meshes; no empty trailing block.
  REQUIRE(source.size() == meshes.size());
  REQUIRE(domain_ids.size() == meshes.size());

  const auto n_faces = index_t(box.polygons().faces().size());
  for (std::size_t c = 0; c < meshes.size(); ++c) {
    auto block = source[c];
    // One source face id per emitted face, in emitted-face order.
    REQUIRE(std::size_t(block.size()) == meshes[c].polygons().faces().size());
    const auto dom = domain_ids[c];
    for (auto f : block) {
      REQUIRE(f >= index_t(0));
      REQUIRE(f < n_faces);
      // Original face f bounds this domain on at least one of its two sides.
      auto sides = labels.labels[f];
      REQUIRE((sides[0] == dom || sides[1] == dom));
    }
  }

  // The plain overload is unchanged and agrees on meshes/labels.
  auto [meshes2, domain_ids2] = tf::split_into_domains(box.polygons(), labels);
  REQUIRE(meshes2.size() == meshes.size());
  REQUIRE(domain_ids2.size() == domain_ids.size());
}

TEST_CASE("split_into_domains return_source_ids (empty input)",
          "[reindex][domains][source_ids]") {
  // No faces: offsets is empty, so the source-buffer trim must not reallocate
  // to size 1 and read an uninitialized offset. Everything comes back empty.
  using index_t = int;
  using real_t = float;
  tf::polygons_buffer<index_t, real_t, 3, 3> empty;
  auto labels = tf::make_domain_labels(empty.polygons());

  auto [meshes, domain_ids, source] =
      tf::split_into_domains(empty.polygons(), labels, tf::return_source_ids);
  REQUIRE(meshes.size() == 0);
  REQUIRE(domain_ids.size() == 0);
  REQUIRE(source.size() == 0);
}

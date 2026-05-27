// Working with arrangements: domains, selection, and signed cuts.
//
// Builds two overlapping cubes and a bisecting plane, computes the
// arrangement, partitions it into watertight bounded domains, and uses
// the plane's winding to split the resulting volumes into "above" and
// "below" sets.

#include <trueform/trueform.hpp>

#include <algorithm>
#include <iostream>
#include <string>

int main() {
  using Index = int;
  using Real = float;

  // -------------------------------------------------------------------
  // 1. Geometry: two overlapping cubes (closed) and a bisecting plane
  //    (open). Cube 0 sits at x=-0.5, cube 1 at x=+0.5, plane lies on
  //    z=0 with stored normal +Z.
  // -------------------------------------------------------------------
  auto cube0 = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto cube1 = tf::make_box_mesh<Index>(Real(2), Real(2), Real(2));
  auto plane = tf::make_plane_mesh<Index>(Real(4), Real(4));

  auto f0 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<Real, 3>{Real(-0.5), Real(0), Real(0)}));
  auto f1 = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<Real, 3>{Real(0.5), Real(0), Real(0)}));
  auto fid = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<Real, 3>{Real(0), Real(0), Real(0)}));

  auto p0 = cube0.polygons() | tf::tag(f0);   // tag 0
  auto p1 = cube1.polygons() | tf::tag(f1);   // tag 1
  auto p_knife = plane.polygons() | tf::tag(fid);  // tag 2 (the knife)

  decltype(p0) forms[] = {p0, p1, p_knife};

  // -------------------------------------------------------------------
  // 2. Arrangement: split every face at every intersection and merge
  //    into a single triangle mesh. tag_labels[f] records which input
  //    operand face f came from (0, 1, or 2 here).
  // -------------------------------------------------------------------
  auto [arr_raw, tag_labels_raw, face_labels_raw] =
      tf::make_mesh_arrangements(tf::make_range(forms, forms + 3));

  std::cout << "=== Arrangement ===" << std::endl;
  std::cout << "Faces:  " << arr_raw.faces().size() << std::endl;
  std::cout << "Points: " << arr_raw.points().size() << std::endl;

  // -------------------------------------------------------------------
  // 3. Clean coincident vertices. The per-face tag_labels array goes
  //    stale when cleaning drops duplicate faces, so we reindex it
  //    through the face index map (kept_ids gives the surviving old
  //    ids in new order).
  // -------------------------------------------------------------------
  auto [arr, face_im, point_im] =
      tf::cleaned(arr_raw.polygons(), tf::epsilon<Real>, tf::return_index_map);
  auto tag_labels = tf::reindexed(tf::make_range(tag_labels_raw), face_im);

  std::cout << "\n=== Cleaned ===" << std::endl;
  std::cout << "Faces:  " << arr.faces().size() << std::endl;
  std::cout << "Points: " << arr.points().size() << std::endl;

  // -------------------------------------------------------------------
  // 4. Domain labels. ignore_open_fragments parks the plane's open
  //    outer ring at the sentinel; exclude_outer_shell folds the
  //    unbounded universe into the same sentinel. What remains is the
  //    bounded interior domains.
  // -------------------------------------------------------------------
  auto dl = tf::make_domain_labels(arr.polygons(),
                                   tf::domain_config::ignore_open_fragments |
                                   tf::domain_config::exclude_outer_shell);
  std::cout << "\n=== Domain labels ===" << std::endl;
  std::cout << "Bounded domains: " << dl.n_domains << std::endl;

  // -------------------------------------------------------------------
  // 5. Split into per-domain watertight outward-oriented submeshes.
  //    comp_labels[i] is the domain id of volumes[i].
  // -------------------------------------------------------------------
  auto [volumes, comp_labels] = tf::split_into_domains(arr.polygons(), dl);
  std::cout << "Volumes extracted: " << volumes.size() << std::endl;

  // -------------------------------------------------------------------
  // 6. Split the volumes by signed side of the knife.
  //
  //    labels[f, 0] = domain containing face f with reversed winding
  //                   (the side f's stored normal points INTO).
  //    labels[f, 1] = domain containing face f with forward winding.
  //
  //    The knife has tag 2 with stored normal +Z, so slot 0 yields the
  //    "above" domain and slot 1 yields the "below" domain. Across all
  //    interior knife faces we collect the unique domain ids — there
  //    can be multiple per side when the knife cuts through several
  //    closed regions (here: cube0-only, intersection, cube1-only).
  // -------------------------------------------------------------------
  tf::buffer<Index> above_ids, below_ids;
  tf::generic_generate(
      tf::zip(tag_labels, dl.labels), std::tie(above_ids, below_ids),
      [&](auto elem, auto &buffers) {
        auto [tag, sides] = elem;
        if (tag != Index(2)) return;
        auto above = sides[0];
        auto below = sides[1];
        if (above >= dl.n_domains || below >= dl.n_domains) return;
        auto &[ab, be] = buffers;
        if (std::find(ab.begin(), ab.end(), above) == ab.end())
          ab.push_back(above);
        if (std::find(be.begin(), be.end(), below) == be.end())
          be.push_back(below);
      });
  auto sort_unique = [](tf::buffer<Index> &b) {
    tbb::parallel_sort(b.begin(), b.end());
    b.erase_till_end(std::unique(b.begin(), b.end()));
  };
  sort_unique(above_ids);
  sort_unique(below_ids);

  std::cout << "\n=== Signed side of the knife ===" << std::endl;
  std::cout << "Above (+normal): " << above_ids.size() << " volumes" << std::endl;
  std::cout << "Below (-normal): " << below_ids.size() << " volumes" << std::endl;

  // -------------------------------------------------------------------
  // 7. Write each side's volumes to disk for visualisation.
  // -------------------------------------------------------------------
  tf::buffer<Index> domain_to_idx;
  domain_to_idx.allocate_and_initialize(
      static_cast<std::size_t>(dl.n_domains), Index(-1));
  tf::invert_map_with_nones(comp_labels, domain_to_idx, Index(-1));

  auto write_side = [&, &vols = volumes](const tf::buffer<Index> &ids,
                                          const char *prefix) {
    for (std::size_t k = 0; k < ids.size(); ++k) {
      auto v_idx = domain_to_idx[static_cast<std::size_t>(ids[k])];
      const auto &vol = vols[static_cast<std::size_t>(v_idx)];
      auto fname = std::string(prefix) + "_" + std::to_string(k) + ".stl";
      tf::write_stl(vol.polygons(), fname);
      std::cout << "  wrote " << fname << " (faces=" << vol.faces().size()
                << ", closed=" << tf::is_closed(vol.polygons())
                << ", manifold=" << tf::is_manifold(vol.polygons()) << ")"
                << std::endl;
    }
  };
  write_side(above_ids, "above");
  write_side(below_ids, "below");

  return 0;
}

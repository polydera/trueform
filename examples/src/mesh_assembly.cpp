#include "cylinder.hpp"

#include <trueform/trueform.hpp>

#include <iostream>

int main() {
  float radius = 1;
  // Create an open tube (cylinder without caps)
  // Each quad has 4 unique vertices - no sharing between faces
  tf::polygons_buffer<int, float, 3, 4> tube =
      examples::make_tube(radius, 4.0f, 16);

  std::cout << "=== Tube (uncleaned) ===" << std::endl;
  std::cout << "Points: " << tube.points().size() << std::endl;
  std::cout << "Faces: " << tube.polygons().size() << std::endl;

  // Without shared vertices, each face is topologically isolated.
  // Every edge appears as a boundary because no two faces share an edge.
  tf::offset_block_buffer<int, int> tube_boundaries =
      tf::make_boundary_paths(tube.polygons());
  std::cout << "Boundary loops: " << tube_boundaries.size() << std::endl;

  // Clean the tube - merges coincident vertices within epsilon tolerance.
  // This creates shared vertices, establishing topological connectivity.
  tf::polygons_buffer<int, float, 3, 4> tube_cleaned =
      tf::cleaned(tube.polygons(), tf::epsilon<float>);

  std::cout << "\n=== Tube (cleaned) ===" << std::endl;
  std::cout << "Points: " << tube_cleaned.points().size() << std::endl;
  std::cout << "Faces: " << tube_cleaned.polygons().size() << std::endl;

  // Now faces share vertices and edges. Only the top and bottom rings
  // remain as boundaries (edges belonging to exactly one face).
  tf::offset_block_buffer<int, int> tube_cleaned_boundaries =
      tf::make_boundary_paths(tube_cleaned.polygons());
  std::cout << "Boundary loops: " << tube_cleaned_boundaries.size()
            << std::endl;

  // Create caps at z=0 and z=height
  tf::polygons_buffer<int, float, 3, 3> disk_bottom =
      examples::make_disk(radius, 0.0f, 16);
  tf::polygons_buffer<int, float, 3, 3> disk_top =
      examples::make_disk(radius, 4.0f, 16);

  std::cout << "\n=== Disks ===" << std::endl;
  std::cout << "Bottom disk - Points: " << disk_bottom.points().size()
            << ", Faces: " << disk_bottom.polygons().size() << std::endl;
  std::cout << "Top disk - Points: " << disk_top.points().size()
            << ", Faces: " << disk_top.polygons().size() << std::endl;

  // Concatenate tube + caps. Using uncleaned meshes so coincident vertices
  // exist at the seams between tube and disks.
  // Result type is dynamic_size because we're mixing quads (4) and triangles
  tf::polygons_buffer<int, float, 3, tf::dynamic_size> combined =
      tf::concatenated(tube.polygons(), disk_bottom.polygons(),
                       disk_top.polygons());

  std::cout << "\n=== Combined (uncleaned) ===" << std::endl;
  std::cout << "Points: " << combined.points().size() << std::endl;
  std::cout << "Faces: " << combined.polygons().size() << std::endl;

  // All faces still isolated - every edge is a boundary
  tf::offset_block_buffer<int, int> combined_boundaries =
      tf::make_boundary_paths(combined.polygons());
  std::cout << "Boundary loops: " << combined_boundaries.size() << std::endl;

  // Clean merges all coincident vertices:
  // - Quad edges with adjacent quads (tube connectivity)
  // - Tube boundary edges merge with disk perimeter edges (sealing the caps)
  // - Disk triangles share center and perimeter vertices
  tf::polygons_buffer<int, float, 3, tf::dynamic_size> cylinder =
      tf::cleaned(combined.polygons(), tf::epsilon<float>);

  std::cout << "\n=== Cylinder (cleaned) ===" << std::endl;
  std::cout << "Points: " << cylinder.points().size() << std::endl;
  std::cout << "Faces: " << cylinder.polygons().size() << std::endl;

  // Every edge now belongs to exactly two faces - watertight mesh
  tf::offset_block_buffer<int, int> cylinder_boundaries =
      tf::make_boundary_paths(cylinder.polygons());
  std::cout << "Boundary loops: " << cylinder_boundaries.size() << std::endl;

  // Ensure positive orientation: orients faces consistently, then flips
  // all if signed volume is negative (outward-facing normals)
  tf::ensure_positive_orientation(cylinder.polygons());

  // Triangulate the cylinder - convert quads and n-gons to triangles
  auto triangulated_cylinder = tf::triangulated(cylinder.polygons());

  // Helper to print mesh topology info - we'll use this from now on
  auto print_topology = [](const char *name, const auto &polygons) {
    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << "Points: " << polygons.points().size() << std::endl;
    std::cout << "Faces: " << polygons.size() << std::endl;
    std::cout << "Boundary loops: " << tf::make_boundary_paths(polygons).size()
              << std::endl;
    std::cout << "Non-manifold edges: "
              << tf::make_non_manifold_edges(polygons).size() << std::endl;
  };

  // Triangulation increases face count (quads become 2 triangles each)
  // but preserves topology: still watertight (0 boundaries, 0 non-manifold)
  print_topology("Cylinder (triangulated)", triangulated_cylinder.polygons());

  // Create a cone to place on top of the cylinder.
  // Clean the disk first - make_disk creates isolated triangles (each with its
  // own vertices), cleaning merges coincident vertices into shared ones.
  auto cone_cap = tf::cleaned(disk_top.polygons(), tf::epsilon<float>);

  // Copy for the cone sheet, then modify to create the sloped surface
  auto cone_sheet = cone_cap;

  // Use neighbor search to find the center vertex (closest to centroid)
  tf::aabb_tree<int, float, 3> sheet_tree(cone_sheet.points(),
                                          tf::config_tree(4, 4));
  auto [center_id, _] =
      tf::neighbor_search(cone_sheet.points() | tf::tag(sheet_tree),
                          tf::centroid(cone_sheet.points()));

  // Move center point up to form cone apex
  cone_sheet.points()[center_id][2] += 1.0f;

  // Combine flat cap + cone sheet. After cleaning, perimeter vertices merge
  // while centers stay separate (flat center at z=2, apex at z=3).
  // Result: watertight cone (0 boundaries, 0 non-manifold)
  tf::polygons_buffer<int, float, 3, 3> cone = tf::cleaned(
      tf::concatenated(cone_cap.polygons(), cone_sheet.polygons()).polygons(),
      tf::epsilon<float>);
  tf::ensure_positive_orientation(cone.polygons());

  print_topology("Cone", cone.polygons());

  // BAD APPROACH: Simply concatenating two closed meshes and cleaning.
  // The cylinder's top cap and cone's bottom cap share the same ring of
  // vertices. After cleaning, each edge on that ring belongs to 4 faces
  // (2 from cylinder cap, 2 from cone cap) - these are non-manifold edges!
  tf::polygons_buffer<int, float, 3, 3> merged = tf::cleaned(
      tf::concatenated(triangulated_cylinder.polygons(), cone.polygons())
          .polygons(),
      tf::epsilon<float>);

  print_topology("Merged (concatenate + clean)", merged.polygons());

  // CORRECT APPROACH: Boolean union properly combines two closed meshes.
  // It removes the interior faces where meshes overlap, producing a single
  // watertight manifold result (0 boundaries, 0 non-manifold edges).
  // Result is triangulated: polygons_buffer<int, float, 3, 3>
  auto [pencil, labels0] = tf::make_boolean(
      triangulated_cylinder.polygons(), cone.polygons(), tf::boolean_op::merge);

  print_topology("Pencil (union)", pencil.polygons());

  // Union volume equals sum of parts (cylinder + cone share no interior)
  std::cout << "Volume: " << tf::signed_volume(pencil.polygons())
            << " == " << tf::signed_volume(cylinder.polygons()) << " + "
            << tf::signed_volume(cone.polygons()) << std::endl;

  // Access triangle vertices directly
  // auto [pt0, pt1, pt2] = pencil.polygons().front();

  // Create a Steinmetz solid (bicylinder) by intersecting with a rotated copy.
  // Rotate 90° around X-axis, centered at the cylinder's centroid.
  auto horizontal_cylinder =
      triangulated_cylinder.polygons() |
      tf::tag(
          tf::make_rotation(tf::deg(90.f), tf::axis<0>,
                            tf::centroid(triangulated_cylinder.polygons())));

  // Boolean intersection of vertical pencil and horizontal cylinder.
  // The result is the classic Steinmetz solid - the region inside both shapes.
  auto [bicylinder, labels1] = tf::make_boolean(
      pencil.polygons(), horizontal_cylinder, tf::boolean_op::intersection);

  print_topology("Bicylinder (intersection)", bicylinder.polygons());

  // Compute a distance field from the centroid of the bicylinder.
  // This creates a scalar value per vertex - distance to center.
  auto center = tf::centroid(bicylinder.points());
  tf::buffer<float> scalars;
  scalars.allocate(bicylinder.points().size());
  tf::parallel_transform(bicylinder.points(), scalars, tf::distance_f(center));

  // Find min/max for choosing cut values
  float min_d = *std::min_element(scalars.begin(), scalars.end());
  float max_d = *std::max_element(scalars.begin(), scalars.end());
  std::cout << "\nDistance field range: [" << min_d << ", " << max_d << "]"
            << std::endl;

  // Create isobands - extract only alternating bands to create disconnected
  // shells. 4 cut values create 5 bands (0-4), we take first, middle, last.
  float step = (max_d - min_d) / 5.0f;
  std::array<float, 4> cut_values = {min_d + step, min_d + 2 * step,
                                     min_d + 3 * step, min_d + 4 * step};
  std::array<int, 3> selected_bands = {0, 2, 4}; // first, middle, last

  auto [slices, band_labels] = tf::make_isobands(
      bicylinder.polygons(), scalars, tf::make_range(cut_values),
      tf::make_range(selected_bands));

  print_topology("Slices (isobands 0, 2, 4)", slices.polygons());
  std::cout << "Selected bands: first, middle, last" << std::endl;

  tf::write_stl(bicylinder.polygons(), "bicylinder.stl");
  tf::write_stl(slices.polygons(), "slices.stl");

  // The bicylinder's symmetry causes some bands to split into separate
  // components - selecting 3 bands yields 5 connected regions.
  auto [component_labels, n_components] =
      tf::make_manifold_edge_connected_component_labels(slices.polygons());

  std::cout << "\nConnected components: " << n_components << std::endl;

  // Split the mesh into separate component meshes.
  auto [components, comp_ids] =
      tf::split_into_components(slices.polygons(), component_labels);

  std::cout << "Split into " << components.size() << " meshes:" << std::endl;
  for (const auto &[i, component] : tf::enumerate(components)) {
    std::cout << "  Component " << comp_ids[i] << ": "
              << component.polygons().size() << " faces, "
              << tf::make_boundary_paths(component.polygons()).size()
              << " boundary loops" << std::endl;

    // Write each component to a separate file
    tf::write_stl(component.polygons(),
                  "component_" + std::to_string(comp_ids[i]) + ".stl");
  }
  return 0;
}

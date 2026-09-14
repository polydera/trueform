#include <trueform/core/polygons_buffer.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/elementwise.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/core/nd_array.hpp>
#include <trueform/cpp/core/nd_array_creation.hpp>
#include <trueform/cpp/geometry/area.hpp>
#include <trueform/cpp/geometry/make_sphere_mesh.hpp>
#include <trueform/cpp/geometry/volume.hpp>

#include <cmath>
#include <iostream>

int main() {
  using real = double;
  using index = tf::cpp::default_index_t;

  // The geometry is the caller's. Core's own buffer owns it; this facade
  // invents no storage, and a caller who holds VTK's arrays or a mapping holds
  // them just as well.
  auto storage = tf::cpp::make_sphere_mesh(real{2}, 16, 32);

  // The cache is the caller's too. It computes nothing the geometry does not
  // determine — it remembers — so dropping it loses only time.
  tf::cpp::cache<index, real> cache;

  // The mesh is the assembly of the two, placed by this instance's frame
  // (identity when none is given). Every entry takes it.
  const tf::cpp::mesh<index, real> sphere{storage.faces(), storage.points(),
                                          cache};
  const auto area = tf::cpp::area(sphere);
  const auto volume = tf::cpp::volume(sphere);

  // The caller states what changed; the cache remembers what it built for, so
  // the structures a moved point invalidates rebuild on their next ask and the
  // rest stand. A mesh is one reading, so a stated change is read by assembling
  // again.
  for (auto point : storage.points())
    point[2] *= real{2};
  cache.points_changed();
  const tf::cpp::mesh<index, real> stretched{storage.faces(), storage.points(),
                                             cache};
  const auto stretched_volume = tf::cpp::volume(stretched);

  // Arrays are the other half of the surface: what entries hand back and what
  // they take. Elementwise operations compose without touching what they read.
  const auto samples = tf::cpp::linspace<real>(real{0}, real{1}, 5);
  const auto scaled = tf::cpp::mul_scalar(samples, real{2});
  const auto shifted = tf::cpp::add_scalar(scaled, real{1});

  const bool assembly_reads_the_callers_storage =
      sphere.number_of_points() == storage.points_buffer().size() &&
      sphere.number_of_faces() == storage.size();
  const bool the_change_was_heard = stretched_volume > volume * real{1.9};
  const bool arrays_compose =
      samples.length() == 5 &&
      std::abs(shifted[4] - (samples[4] * real{2} + real{1})) < real{1e-12};

  std::cout << "UV sphere: " << sphere.number_of_points() << " vertices, "
            << sphere.number_of_faces() << " triangles\n"
            << "area " << area << ", volume " << volume << "; after stating "
            << "the points moved: " << stretched_volume << '\n'
            << "arrays: " << samples.length() << " samples, last " << shifted[4]
            << '\n';

  return assembly_reads_the_callers_storage && the_change_was_heard &&
                 arrays_compose
             ? 0
             : 1;
}

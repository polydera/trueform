#include <trueform/core/polygons_buffer.hpp>
#include <trueform/cpp/core/build_tree.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/csg/async/outer_shell.hpp>
#include <trueform/cpp/geometry/make_box_mesh.hpp>

#include <future>
#include <iostream>
#include <type_traits>

int main() {
  using real = double;
  using index = tf::cpp::default_index_t;

  const auto storage = tf::cpp::make_box_mesh(real{2}, real{3}, real{4});
  tf::cpp::cache<index, real> cache;
  const tf::cpp::mesh<index, real> box{storage.faces(), storage.points(),
                                       cache};

  // Using the cache under async? Prebuild what needs prebuilding: a build verb
  // fills on this thread, so the worker only reads what is already there.
  tf::cpp::build_tree(box);

  auto pending_shell = tf::cpp::async::outer_shell(box);
  static_assert(
      std::is_same_v<decltype(pending_shell),
                     std::future<tf::polygons_buffer<index, real, 3, 3>>>);

  // A job carries the mesh AS IT STANDS: one coherent reading of memory this
  // scope keeps alive until the future completes, which is the same borrow law
  // a synchronous call obeys. The result is storage of its own.
  const auto shell = pending_shell.get();
  std::cout << "async outer shell: " << shell.points_buffer().size()
            << " vertices, " << shell.size() << " triangles\n";

  return shell.size() > 0 ? 0 : 1;
}

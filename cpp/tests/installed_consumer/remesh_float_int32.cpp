#include <trueform/cpp/remesh/decimated.hpp>
#include <trueform/cpp/remesh/isotropic_remeshed.hpp>
#include <trueform/cpp/remesh/simplified.hpp>

#include "axis_shard_test_utils.hpp"

#include <cstdint>
#include <type_traits>

int main() {
  using real_type = float;
  using index_type = std::int32_t;
  const auto owned =
      trueform_installed_test::negative_tetrahedron<index_type, real_type>();
  const auto value = owned.mesh();

  tf::decimate_config<real_type> decimate;
  decimate.parallel = false;
  tf::isotropic_remesh_config<real_type> isotropic(real_type{1}, 0);
  isotropic.parallel = false;
  tf::simplify_config<real_type> simplify;
  simplify.iterations = 0;
  simplify.parallel = false;

  const auto decimated = tf::cpp::decimated<index_type, real_type>(
      value, real_type{1}, decimate);
  const auto remeshed =
      tf::cpp::isotropic_remeshed<index_type, real_type>(value, isotropic);
  const auto simplified =
      tf::cpp::simplified<index_type, real_type>(value, simplify);
  using result_type = tf::cpp::remesh_result<index_type, real_type>;
  static_assert(std::is_same_v<decltype(decimated), const result_type>);
  return decimated.mesh.size() != 0 && remeshed.mesh.size() != 0 &&
                 simplified.mesh.size() != 0
             ? 0
             : 1;
}

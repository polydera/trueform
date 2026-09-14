#include "spatial_float_int64_2d_test_utils.hpp"

#include <trueform/cpp/spatial/intersects.hpp>

int main() {
  const auto owned = spatial_float_int64_2d_test::triangle_mesh();
  const auto form = owned.mesh();
  const auto query = spatial_float_int64_2d_test::point();
  const auto forward = tf::cpp::intersects(form, query);
  const auto reverse = tf::cpp::intersects(query, form);
  return forward.is_scalar() && forward.scalar() && reverse.is_scalar() &&
                 reverse.scalar()
             ? 0
             : 1;
}

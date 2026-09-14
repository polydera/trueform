#include <trueform/cpp/core/nd_array.hpp>
#include <trueform/cpp/spatial/primitive.hpp>

// Frozen declarations from before mesh-normal overloads and their archive
// shards existed. Do not include geometry/normals.hpp here: these addresses
// must retain the historical runtime-primitive symbols exactly.
namespace tf::cpp {

template <typename Real>
auto normals(const primitive<Real, 3> &) -> nd_array<Real>;

} // namespace tf::cpp

namespace {

template <typename Pointer> auto retain(Pointer pointer) -> bool {
  volatile auto retained = pointer;
  return retained != nullptr;
}

template <typename Real> auto retain_legacy_primitive_normal_symbol() -> bool {
  using function_type =
      tf::cpp::nd_array<Real> (*)(const tf::cpp::primitive<Real, 3> &);
  return retain(static_cast<function_type>(&tf::cpp::normals<Real>));
}

} // namespace

int main() {
  return retain_legacy_primitive_normal_symbol<float>() &&
                 retain_legacy_primitive_normal_symbol<double>()
             ? 0
             : 1;
}

// The test fixtures, reachable by name for a driver that takes one on its
// command line.
#pragma once

#include "../../tests/common/volume_generators.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

namespace tf {
namespace test {

/// @brief Call `f(field)` with the named field, or return false.
///
/// Every fixture `volume_generators.hpp` carries is reachable here, the
/// parameterised ones by the number their name ends in — `wedge30`, `cone20`,
/// `rounded_box0.05` — so a driver scores the ladder the research states.
template <typename F> auto with_named_field(const char *name, F &&f) -> bool {
  namespace vf = tf::test::volume_field;
  if (std::strcmp(name, "sphere") == 0)
    return f(vf::sphere{}), true;
  if (std::strcmp(name, "box") == 0)
    return f(vf::box{}), true;
  if (std::strcmp(name, "gear") == 0)
    return f(vf::gear{}), true;
  if (std::strcmp(name, "slab") == 0)
    return f(vf::slab{}), true;
  if (std::strcmp(name, "bounded_slab") == 0)
    return f(vf::bounded_slab{}), true;
  if (std::strcmp(name, "gyroid") == 0)
    return f(vf::gyroid{}), true;
  if (std::strcmp(name, "bounded_gyroid") == 0)
    return f(vf::bounded_gyroid{}), true;
  if (std::strcmp(name, "noise") == 0)
    return f(vf::noise{}), true;
  if (std::strcmp(name, "octahedron") == 0)
    return f(vf::octahedron{}), true;
  if (std::strcmp(name, "cylinder") == 0)
    return f(vf::cylinder{}), true;
  if (std::strcmp(name, "box_cylinder") == 0)
    return f(vf::box_cylinder{}), true;
  if (std::strncmp(name, "wedge", 5) == 0) {
    const float angle = float(std::atof(name + 5));
    return f(vf::wedge(angle > 0.f ? angle : 90.f)), true;
  }
  // the ladders the sharpness research is scored on: the cone's half-angle and
  // the rounded box's fillet radius are the parameter each sweeps
  if (std::strncmp(name, "cone", 4) == 0) {
    const float angle = float(std::atof(name + 4));
    return f(vf::cone(angle > 0.f ? angle : 30.f)), true;
  }
  if (std::strncmp(name, "rounded_box", 11) == 0) {
    const float radius = float(std::atof(name + 11));
    return f(vf::rounded_box(radius > 0.f ? radius : 0.f)), true;
  }
  return false;
}

/// @brief The named field sampled onto a cubic grid; an empty volume when the
/// name is not one of them.
inline auto volume_by_name(const std::string &name, int n)
    -> tf::volume_buffer<float> {
  tf::volume_buffer<float> out;
  if (!with_named_field(name.c_str(), [&](const auto &field) {
        out = sampled_volume(field, n);
      }))
    return {};
  return out;
}

} // namespace test
} // namespace tf

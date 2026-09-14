// Timing driver for the volume isosurface entries.
//
// One process = one (n, fixture, threads, refine) cell. The total is timed
// first, on a build that records nothing; the pass table follows from its own
// reps, so a pass's share is priced against the same work. Peak RSS is
// reported, output determinism is proven, and — with refinement on — how far
// the refit actually moved vertices.
//
//   ./bench_isosurface --n 128 --fixture box --threads 8 --refine 1
//                      [--stabilizer 0.01] [--warm 3] [--reps 7] [--quiet]
#include "volume_fields.hpp"

#include <trueform/volume/impl/flying_dc.hpp>
#include <trueform/volume/make_isosurface.hpp>

#include <trueform/core/algorithm/parallel_for_each.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/views/sequence_range.hpp>

#include <tbb/global_control.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

using mesh_t = tf::polygons_buffer<int, float, 3, 3>;

auto now() { return std::chrono::steady_clock::now(); }
auto ms(std::chrono::steady_clock::time_point a,
        std::chrono::steady_clock::time_point b) -> double {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

auto median_of(std::vector<double> v) -> double {
  std::sort(v.begin(), v.end());
  const auto n = v.size();
  return n == 0 ? 0.0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

auto peak_rss_mib() -> double {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return double(ru.ru_maxrss) / (1024.0 * 1024.0);
#else
  return double(ru.ru_maxrss) / 1024.0;
#endif
}

auto identical(const mesh_t &a, const mesh_t &b) -> bool {
  const auto &pa = a.points_buffer().data_buffer();
  const auto &pb = b.points_buffer().data_buffer();
  const auto &fa = a.faces_buffer().data_buffer();
  const auto &fb = b.faces_buffer().data_buffer();
  return pa.size() == pb.size() && fa.size() == fb.size() &&
         std::memcmp(pa.data(), pb.data(), pa.size() * sizeof(float)) == 0 &&
         std::memcmp(fa.data(), fb.data(), fa.size() * sizeof(int)) == 0;
}

} // namespace

int main(int argc, char **argv) {
  int n = 128;
  int threads = 0;
  int warm = 3;
  int reps = 7;
  int refine = 0;
  int public_entry = 0;
  int flying_edges = 0;
  double stabilizer = 0.01;
  bool quiet = false;
  std::string fixture = "box";
  for (int i = 1; i < argc; ++i) {
    const auto arg = std::string(argv[i]);
    const auto next = [&]() -> const char * {
      return i + 1 < argc ? argv[++i] : "";
    };
    if (arg == "--n")
      n = std::atoi(next());
    else if (arg == "--threads")
      threads = std::atoi(next());
    else if (arg == "--warm")
      warm = std::atoi(next());
    else if (arg == "--reps")
      reps = std::atoi(next());
    else if (arg == "--refine")
      refine = std::atoi(next());
    else if (arg == "--public")
      public_entry = 1;
    else if (arg == "--flying-edges")
      flying_edges = 1;
    else if (arg == "--stabilizer")
      stabilizer = std::atof(next());
    else if (arg == "--fixture")
      fixture = next();
    else if (arg == "--quiet")
      quiet = true;
    else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }

  std::unique_ptr<tbb::global_control> limit;
  if (threads > 0)
    limit = std::make_unique<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, std::size_t(threads));

  const auto vol = tf::test::volume_by_name(fixture, n);
  if (vol.samples_buffer().size() == 0) {
    std::fprintf(stderr, "unknown fixture %s\n", fixture.c_str());
    return 2;
  }
  tf::isosurface_config config;
  config.method = tf::isosurface_method::dual_contouring;
  config.refine = refine != 0;
  config.stabilizer = stabilizer;

  tf::volume_detail::dual_contouring<int, float> dc;
  // the public entry, cold: a fresh builder, as a caller gets it
  const auto t_cold = now();
  {
    auto m = tf::make_isosurface(vol.volume(), 0.f, config);
    static_cast<void>(m);
  }
  const double cold_ms = ms(t_cold, now());

  for (int i = 0; i < warm; ++i) {
    auto m = dc.build(vol.volume(), 0.f, config);
    static_cast<void>(m);
  }

  std::vector<double> totals;
  mesh_t timed_mesh;
  double best_total = 1e300;
  if (flying_edges)
    config.method = tf::isosurface_method::flying_edges;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = now();
    // the public entry builds a fresh extractor, which is what a caller pays
    auto m = public_entry ? tf::make_isosurface(vol.volume(), 0.f, config)
                          : dc.build(vol.volume(), 0.f, config);
    const double total = ms(t0, now());
    totals.push_back(total);
    if (total < best_total) {
      best_total = total;
      timed_mesh = std::move(m);
    }
  }

  const bool deterministic = identical(timed_mesh, dc.build(vol.volume(), 0.f, config));

  // the pass table: its own reps, so the total above priced no clock
  std::vector<double> pass_reps[tf::volume_detail::k_dc_pass_count];
  if (!flying_edges) {
    dc.record_pass_times(true);
    for (int i = 0; i < reps; ++i) {
      auto m = dc.build(vol.volume(), 0.f, config);
      static_cast<void>(m);
      for (int p = 0; p < tf::volume_detail::k_dc_pass_count; ++p)
        pass_reps[p].push_back(dc.pass_times()[std::size_t(p)]);
    }
    dc.record_pass_times(false);
  }

  // Peak RSS of the extraction alone: the displacement check below builds a
  // second extractor and would double it.
  const double rss = peak_rss_mib();

  // What refinement actually moved, against the same build with pass 6 off.
  double moved_frac = 0, moved_mean = 0, moved_max = 0;
  if (refine > 0) {
    tf::volume_detail::dual_contouring<int, float> plain;
    auto plain_config = config;
    plain_config.refine = false;
    auto base_mesh = plain.build(vol.volume(), 0.f, plain_config);
    const auto a = base_mesh.points();
    const auto b = timed_mesh.points();
    if (a.size() == b.size()) {
      std::size_t moved = 0;
      double sum = 0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        const double dx = double(b[i][0]) - a[i][0];
        const double dy = double(b[i][1]) - a[i][1];
        const double dz = double(b[i][2]) - a[i][2];
        const double d =
            std::sqrt(dx * dx + dy * dy + dz * dz) / double(vol.spacing()[0]);
        if (d > 0) {
          ++moved;
          sum += d;
          moved_max = std::max(moved_max, d);
        }
      }
      moved_frac = a.size() ? double(moved) / double(a.size()) : 0.0;
      moved_mean = moved ? sum / double(moved) : 0.0;
    } else {
      moved_frac = -1;
    }
  }

  std::printf("DC fixture=%s n=%d threads=%d refine=%d stab=%.3f "
              "med=%.3f min=%.3f cold=%.3f verts=%lld tris=%lld out_MiB=%.2f "
              "rss_MiB=%.1f deterministic=%d moved_frac=%.4f "
              "moved_mean_sp=%.4f moved_max_sp=%.3f\n",
              fixture.c_str(), n, threads, refine, stabilizer,
              median_of(totals),
              *std::min_element(totals.begin(), totals.end()), cold_ms,
              (long long)timed_mesh.points().size(),
              (long long)timed_mesh.polygons().size(),
              double(timed_mesh.points().size() * 3 * 4 +
                     timed_mesh.polygons().size() * 3 * 4) /
                  1048576.0,
              rss, int(deterministic), moved_frac, moved_mean, moved_max);

  if (!flying_edges) {
    double table_total = 0;
    for (int p = 0; p < tf::volume_detail::k_dc_pass_count; ++p)
      table_total += median_of(pass_reps[std::size_t(p)]);
    std::printf("   passes (median ms, sum=%.3f):", table_total);
    for (int p = 0; p < tf::volume_detail::k_dc_pass_count; ++p) {
      const double t = median_of(pass_reps[std::size_t(p)]);
      std::printf(" %s=%.3f(%.1f%%)",
                  tf::volume_detail::dc_pass_name(tf::volume_detail::dc_pass(p)),
                  t, table_total > 0 ? 100.0 * t / table_total : 0.0);
    }
    std::printf("\n");
  }

  if (!quiet) {
    std::printf("   totals:");
    for (auto t : totals)
      std::printf(" %.3f", t);
    std::printf("\n");
  }
  return deterministic ? 0 : 1;
}

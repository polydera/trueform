// Timing driver for the Delaunay tetrahedralization.
//
// One process = one (n, fixture, threads, policy) cell. The sites are lattice
// points drawn from a fixed seed; the build is timed cold, then over warm and
// timed reps on one reused builder, as a caller holding it pays. The published
// stream is proven byte-identical to the serial policy's before a time is
// reported, and peak RSS is reported beside it.
//
//   ./bench_tetrahedralization --n 1000000 --fixture cloud --threads 16
//                              [--serial] [--warm 2] [--reps 7] [--quiet]
//
// The ladder is one process per cell, run one at a time:
//
//   for n in 10000 100000 1000000 4000000; do
//     ./bench_tetrahedralization --n $n --serial
//     for t in 1 4 8 16; do ./bench_tetrahedralization --n $n --threads $t; done
//   done
//
// Fixtures: cloud (uniform in a cube), sphere (on a sphere's surface, every
// site on the hull), grid (a cubic grid, every cell's sphere cospherical).
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/topology/cdt/delaunay_execution_policy.hpp>
#include <trueform/topology/delaunay_tetrahedralizer.hpp>
#include <trueform/topology/tetrahedralization_stats.hpp>

#include <tbb/task_arena.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

using parallel_dt = tf::delaunay_tetrahedralizer<int, tf::exact::int32>;
using serial_dt = tf::delaunay_tetrahedralizer<
    int, tf::exact::int32, tf::exact::int32,
    tf::topology::cdt::serial_delaunay_execution_policy>;

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

auto make_sites(const std::string &fixture, int n)
    -> std::vector<std::int32_t> {
  std::vector<std::int32_t> flat;
  flat.reserve(3 * std::size_t(n));
  std::mt19937_64 rng(20261006u);
  const double span = double(std::int32_t(1) << 28);
  if (fixture == "cloud") {
    std::uniform_int_distribution<std::int32_t> axis(-(1 << 28), 1 << 28);
    for (int i = 0; i < 3 * n; ++i)
      flat.push_back(axis(rng));
  } else if (fixture == "sphere") {
    std::normal_distribution<double> axis(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
      const double x = axis(rng), y = axis(rng), z = axis(rng);
      const double scale = span / std::sqrt(x * x + y * y + z * z);
      flat.push_back(std::int32_t(std::llround(x * scale)));
      flat.push_back(std::int32_t(std::llround(y * scale)));
      flat.push_back(std::int32_t(std::llround(z * scale)));
    }
  } else if (fixture == "grid") {
    const int side = int(std::ceil(std::cbrt(double(n))));
    for (int i = 0; i < n; ++i) {
      flat.push_back(std::int32_t(64 * (i % side)));
      flat.push_back(std::int32_t(64 * ((i / side) % side)));
      flat.push_back(std::int32_t(64 * (i / (side * side))));
    }
  }
  return flat;
}

template <typename DT> auto published_stream(const DT &dt) -> std::vector<int> {
  std::vector<int> stream;
  stream.reserve(8 * dt.n_tets());
  for (auto cell : dt.tets())
    for (std::size_t slot = 0; slot < 4; ++slot)
      stream.push_back(cell[slot]);
  for (auto links : dt.neighbors())
    for (std::size_t slot = 0; slot < 4; ++slot)
      stream.push_back(links[slot]);
  return stream;
}

struct timed_builds {
  double cold = 0;
  std::vector<double> totals;
  std::vector<int> stream;
  std::size_t sites = 0, tets = 0;
  tf::tetrahedralization_stats stats;
  bool built = true;
};

template <typename DT, typename Points>
auto time_builds(const Points &points, int warm, int reps) -> timed_builds {
  timed_builds out;
  const auto t_cold = now();
  {
    DT cold;
    out.built = cold.build(points);
  }
  out.cold = ms(t_cold, now());
  DT dt;
  for (int i = 0; i < warm; ++i)
    out.built = dt.build(points) && out.built;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = now();
    out.built = dt.build(points) && out.built;
    out.totals.push_back(ms(t0, now()));
  }
  out.stream = published_stream(dt);
  out.sites = dt.n_sites();
  out.tets = dt.n_tets();
  out.stats = dt.stats();
  return out;
}

} // namespace

int main(int argc, char **argv) {
  int n = 100000;
  int threads = 0;
  int warm = 2;
  int reps = 7;
  bool serial = false;
  bool quiet = false;
  std::string fixture = "cloud";
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
    else if (arg == "--serial")
      serial = true;
    else if (arg == "--fixture")
      fixture = next();
    else if (arg == "--quiet")
      quiet = true;
    else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  reps = std::max(reps, 1);

  const auto flat = make_sites(fixture, n);
  if (flat.empty()) {
    std::fprintf(stderr, "unknown fixture %s\n", fixture.c_str());
    return 2;
  }
  const auto points = tf::make_points<3>(tf::make_range(flat));

  // The arena, not a global limit, is what the build reads its worker count
  // and its stream count from.
  tbb::task_arena arena(threads > 0 ? threads : tbb::task_arena::automatic);
  const auto timed = arena.execute([&] {
    return serial ? time_builds<serial_dt>(points, warm, reps)
                  : time_builds<parallel_dt>(points, warm, reps);
  });
  // Peak RSS of the timed builds alone: the reference below builds again.
  const double rss = peak_rss_mib();

  serial_dt reference;
  const bool deterministic = timed.built && reference.build(points) &&
                             published_stream(reference) == timed.stream;

  std::printf("DT3 fixture=%s n=%d threads=%d policy=%s med=%.3f min=%.3f "
              "cold=%.3f sites=%zu tets=%zu tets_per_site=%.3f "
              "rss_MiB=%.1f bands=%zu parallel=%zu restarts=%zu "
              "compactions=%zu deterministic=%d\n",
              fixture.c_str(), n, threads, serial ? "serial" : "parallel",
              median_of(timed.totals),
              *std::min_element(timed.totals.begin(), timed.totals.end()),
              timed.cold, timed.sites, timed.tets,
              timed.sites ? double(timed.tets) / double(timed.sites) : 0.0,
              rss, timed.stats.parallel_bands, timed.stats.parallel_insertions,
              timed.stats.pool_restarts, timed.stats.compactions,
              int(deterministic));
  if (!quiet) {
    std::printf("   totals:");
    for (auto t : timed.totals)
      std::printf(" %.3f", t);
    std::printf("\n");
  }
  return deterministic ? 0 : 1;
}

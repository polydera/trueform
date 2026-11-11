// Timing driver for the tet mesher.
//
// tf::make_tet_mesh over a list of surfaces, at each worker count given, the
// best of `--reps` calls, one row per model: model | status | cells | volume
// | ms per worker count, then a sum line. The columns are fixed, so two runs
// diff. Model files are the caller's: positional paths, or `--list` naming a
// file of paths, one per line (by default the arc's corpus list and its three
// large models, read where they stand on the measuring machine).
//
//   ./bench_tet_mesh --threads 16,1 [--reps 3] [--list corpus.txt] [a.obj ...]
//                    [--refs] [--ref-tables timing.tsv,...]
//
// `--refs` appends the references' timings the arc already measured, at
// their fastest configuration: HXT (raw, no refinement) at 16 and at 1
// workers, and TetGen's faster valid arm, single threaded. They are read from
// the arc's timing tables (`--ref-tables`, by default the ones it measured
// beside its references), keyed by the model's file stem, the run's
// minimum where a table states one and its median otherwise; a time whose
// product the table did not validate carries a `*`, and a model a table does
// not name reads `-`.
#include <trueform/geometry/triangulated.hpp>
#include <trueform/io/read_obj.hpp>
#include <trueform/tetmesh/make_tet_mesh.hpp>

#include <tbb/task_arena.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char *const default_list =
    "/Users/ziga/trueform-scratch-dt-kernel/probes/inline/step3/corpus75.txt";
const char *const default_large[] = {
    "/Users/ziga/trueform-scratch-dt-kernel/probes/bench_tet_refs/large/in/"
    "igea_x16.obj",
    "/Users/ziga/trueform-scratch-dt-kernel/probes/bench_tet_refs/large/in/"
    "fertility_x64.obj",
    "/Users/ziga/trueform-scratch-dt-kernel/probes/bench_tet_refs/large/in/"
    "lucy_x16.obj"};
const char *const default_refs[] = {
    "/Users/ziga/trueform-scratch-dt-kernel/probes/bench_tet_refs/timing.tsv",
    "/Users/ziga/trueform-scratch-dt-kernel/probes/bench_tet_refs/large/"
    "timing.tsv"};

auto split(const std::string &text, char by) -> std::vector<std::string> {
  std::vector<std::string> parts;
  std::string part;
  std::istringstream in(text);
  while (std::getline(in, part, by))
    parts.push_back(part);
  return parts;
}

auto stem_of(const std::string &path) -> std::string {
  const auto slash = path.find_last_of('/');
  auto name = slash == std::string::npos ? path : path.substr(slash + 1);
  const auto dot = name.find_last_of('.');
  return dot == std::string::npos ? name : name.substr(0, dot);
}

auto status_name(tf::tet_mesh_status status) -> const char * {
  switch (status) {
  case tf::tet_mesh_status::meshed:
    return "meshed";
  case tf::tet_mesh_status::refused:
    return "refused";
  case tf::tet_mesh_status::unwound:
    return "unwound";
  case tf::tet_mesh_status::rank_deficient:
    return "rank_deficient";
  case tf::tet_mesh_status::capacity:
    return "capacity";
  case tf::tet_mesh_status::open:
    return "open";
  }
  return "?";
}

struct run {
  tf::tet_mesh_status status = tf::tet_mesh_status::meshed;
  std::size_t cells = 0;
  double volume = 0;
  double ms = 0;
};

auto mesh_best_of(const tf::polygons_buffer<int, float, 3, 3> &surface,
                  int threads, int reps) -> run {
  tbb::task_arena arena(threads > 0 ? threads : tbb::task_arena::automatic);
  return arena.execute([&] {
    run best;
    best.ms = 1e300;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto [tets, status] = tf::make_tet_mesh(surface.polygons());
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      if (ms < best.ms) {
        best.ms = ms;
        best.status = status;
        best.cells = std::size_t(tets.size());
        double volume = 0;
        for (const auto &t : tets.tetrahedra()) {
          const auto a = t[1] - t[0], b = t[2] - t[0], c = t[3] - t[0];
          volume += double(tf::dot(tf::cross(a, b), c)) / 6;
        }
        best.volume = volume;
      }
    }
    return best;
  });
}

// One reference cell: a time and whether the table validated its product.
struct ref_time {
  double ms = -1;
  bool valid = false;
};

// Per model, the references' columns of the timing tables.
using ref_table = std::map<std::string, std::map<std::string, std::string>>;

auto read_refs(const std::vector<std::string> &paths) -> ref_table {
  ref_table table;
  for (const auto &path : paths) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line))
      continue;
    const auto header = split(line, '\t');
    while (std::getline(in, line)) {
      const auto cells = split(line, '\t');
      if (cells.empty())
        continue;
      auto &row = table[cells[0]];
      for (std::size_t i = 1; i < cells.size() && i < header.size(); ++i)
        row[header[i]] = cells[i];
    }
  }
  return table;
}

auto ref_of(const ref_table &table, const std::string &model,
            const std::string &arm) -> ref_time {
  ref_time out;
  const auto row = table.find(model);
  if (row == table.end())
    return out;
  const auto get = [&row](const std::string &key) -> const std::string * {
    const auto at = row->second.find(key);
    return at == row->second.end() || at->second.empty() ? nullptr
                                                          : &at->second;
  };
  const auto *time = get(arm + "_own_min_ms");
  if (!time)
    time = get(arm + "_own_ms");
  if (!time)
    return out;
  out.ms = std::atof(time->c_str());
  const auto *valid = get(arm + "_valid");
  out.valid = !valid || *valid == "1";
  return out;
}

// The faster of two arms, a validated one before one that is not.
auto faster(ref_time a, ref_time b) -> ref_time {
  if (a.ms < 0)
    return b;
  if (b.ms < 0)
    return a;
  if (a.valid != b.valid)
    return a.valid ? a : b;
  return a.ms <= b.ms ? a : b;
}

auto print_ref(ref_time r) -> void {
  if (r.ms < 0)
    std::printf(" | %12s", "-");
  else
    std::printf(" | %11.1f%s", r.ms, r.valid ? " " : "*");
}

} // namespace

int main(int argc, char **argv) {
  std::vector<int> threads{16};
  int reps = 3;
  std::string list;
  std::vector<std::string> paths;
  bool refs = false;
  std::vector<std::string> ref_paths;
  for (int i = 1; i < argc; ++i) {
    const auto arg = std::string(argv[i]);
    const auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : "";
    };
    if (arg == "--threads") {
      threads.clear();
      for (const auto &t : split(next(), ','))
        threads.push_back(std::atoi(t.c_str()));
    } else if (arg == "--reps")
      reps = std::max(1, std::atoi(next().c_str()));
    else if (arg == "--list")
      list = next();
    else if (arg == "--refs")
      refs = true;
    else if (arg == "--ref-tables")
      ref_paths = split(next(), ',');
    else if (!arg.empty() && arg[0] == '-') {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    } else
      paths.push_back(arg);
  }
  if (paths.empty()) {
    std::ifstream in(list.empty() ? default_list : list.c_str());
    std::string line;
    while (std::getline(in, line))
      if (!line.empty())
        paths.push_back(line);
    if (list.empty())
      for (const auto *large : default_large)
        paths.push_back(large);
  }
  if (refs && ref_paths.empty())
    ref_paths.assign(std::begin(default_refs), std::end(default_refs));
  const auto table = read_refs(ref_paths);

  std::printf("%-24s | %-14s | %10s | %18s", "model", "status", "cells",
              "volume");
  for (int t : threads)
    std::printf(" | %9s@%-2d", "ms", t);
  if (refs)
    std::printf(" | %12s | %12s | %12s", "hxt@16", "hxt@1", "tetgen@1");
  std::printf("\n");
  std::vector<double> sums(threads.size(), 0);
  std::size_t cells = 0;
  for (const auto &path : paths) {
    auto surface = tf::read_obj<3>(path);
    if (!surface.faces().size()) {
      // not a triangle file: read at its own arity and triangulated
      const auto whole = tf::read_obj(path);
      if (whole.faces().size())
        surface = tf::triangulated(whole.polygons());
    }
    const auto model = stem_of(path);
    std::printf("%-24s", model.c_str());
    for (std::size_t k = 0; k < threads.size(); ++k) {
      const auto best = mesh_best_of(surface, threads[k], reps);
      if (!k) {
        std::printf(" | %-14s | %10zu | %18.10g", status_name(best.status),
                    best.cells, best.volume);
        cells += best.cells;
      }
      std::printf(" | %12.1f", best.ms);
      sums[k] += best.ms;
    }
    if (refs) {
      print_ref(ref_of(table, model, "hxtraw@16"));
      print_ref(ref_of(table, model, "hxtraw@1"));
      print_ref(faster(ref_of(table, model, "tetgen_p@1"),
                       ref_of(table, model, "tetgen_pY@1")));
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  std::printf("%-24s | %-14s | %10zu | %18s", "sum", "", cells, "");
  for (double s : sums)
    std::printf(" | %12.1f", s);
  std::printf("\n");
  return 0;
}

# Interactive C++ facade + VTK examples

These twelve programs are interactive C++17 ports of the programs in
`python/examples/vtk`. Trueform computation goes through the installed public
`tf::cpp` facade. Visualization uses upstream VTK directly.

The examples deliberately do **not** link `tf::vtk`, include
`<trueform/vtk/...>`, or call header-only Trueform algorithms. Their local
adapters deep-copy the geometry and curves into ordinary `vtkPolyData` objects
so ownership remains explicit on both sides of the boundary.

Each program holds what a caller holds: core's own storage, a cache that
remembers what was built for it, and the placement the instance is drawn at.
`util/vtk_examples.hpp` bundles those three per actor (`mesh_actor_data`) and
`mesh()` is the assembly every `tf::cpp` entry takes.

## Build in the Trueform source tree

Install VTK 9 or newer with its CMake development targets, then configure:

```sh
cmake -S . -B build-vtk-examples \
  -DTF_BUILD_CPP_VTK_EXAMPLES=ON \
  -DTF_BUILD_VTK_INTEGRATION=OFF
cmake --build build-vtk-examples --parallel \
  --target trueform_cpp_vtk_examples
```

`TF_BUILD_VTK_INTEGRATION=OFF` is intentional: these programs are consumers of
`tf::trueform_cpp`, not examples of the legacy integration library. Interactive
executables are not registered with CTest because they wait for a render window
to close.

## Build against an installed Trueform package

First install a package that contains the optional C++ facade:

```sh
cmake -S . -B /tmp/trueform-install-build -DTF_BUILD_CPP=ON
cmake --build /tmp/trueform-install-build --parallel --target trueform_cpp
cmake --install /tmp/trueform-install-build --prefix /tmp/trueform-install
```

The example directory is then standalone:

```sh
cmake -S cpp/examples/vtk -B /tmp/trueform-cpp-vtk-examples \
  -DCMAKE_PREFIX_PATH=/tmp/trueform-install \
  -DTRUEFORM_CPP_VTK_DATA_DIR=/path/to/trueform/source
cmake --build /tmp/trueform-cpp-vtk-examples --parallel
```

When building this directory from its repository location, the data root is
derived from the directory itself. `TRUEFORM_CPP_VTK_DATA_DIR` overrides that
root for a copied example directory or installed-package consumer. Explicit
input paths do not need it.

## Programs

| Executable | Python source | Main interaction |
|---|---|---|
| `trueform_cpp_vtk_alignment` | `alignment.py` | Left-drag source; right-drag rotate; `A` align; `R` randomize |
| `trueform_cpp_vtk_boolean_difference` | `boolean_difference.py` | Drag inputs; `N` randomize orientations |
| `trueform_cpp_vtk_collision` | `collision.py` | Hover and drag a mesh in the 5x5 grid |
| `trueform_cpp_vtk_cross_section` | `cross_section.py` | `N` randomize plane; Ctrl+wheel move section |
| `trueform_cpp_vtk_csg_fracture` | `csg_fracture.py` | Trackball camera; `C` print camera; optional `--all`/`--png` |
| `trueform_cpp_vtk_domains` | `domains.py` | Left/Right cycle domains; Up/Down cycle scenes |
| `trueform_cpp_vtk_domains_from_files` | `domains_from_files.py` | Left/Right cycle domains |
| `trueform_cpp_vtk_intersection_curves` | `intersection_curves.py` | Drag inputs; `N` randomize orientations |
| `trueform_cpp_vtk_isobands` | `isobands.py` | `N` randomize plane; Ctrl+wheel move bands |
| `trueform_cpp_vtk_isocontours` | `isocontours.py` | `N` randomize plane; Ctrl+wheel move contours |
| `trueform_cpp_vtk_remeshing` | `remeshing.py` | Shared trackball camera across three panels |
| `trueform_cpp_vtk_simplify` | `simplify.py` | Shared trackball camera across the 2x3 sweep |

Use `--help` on programs with arguments for exact syntax. `domains` has no
arguments. `csg_fracture --png` is retained because PNG output is part of the
Python example; all other programs are interactive-only.

## Ownership and transformations

VTK actors retain their own `vtkPolyData` and `vtkMatrix4x4`. Draggable examples
write each changed actor matrix back into the instance's placement before
recomputing; the frame is a tag, so nothing the cache built is stale when one
moves. Exact cut examples that operate in local coordinates physically
materialize their center/scale transform into the point storage, matching the
Python programs — and a caller that moves coordinates after a read says
`points_changed()`.

"""
Conan recipe for trueform (header-only core).

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import os
import re
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, load
from conan.errors import ConanInvalidConfiguration


class TrueformConan(ConanFile):
    name = "trueform"
    license = "PolyForm Noncommercial License 1.0.0"
    url = "https://github.com/polydera/trueform"
    homepage = "https://trueform.polydera.com"
    description = "Fast and exact mesh booleans, spatial queries, arrangements, registration, and remeshing. Header-only C++17."
    topics = ("geometry", "mesh", "computational-geometry", "mesh-processing",
              "mesh-boolean", "collision-detection", "csg", "header-only")

    settings = "os", "arch", "compiler", "build_type"
    package_type = "header-library"

    # Mirrors the CMake master switch TF_USE_MIMALLOC: one knob that turns the
    # mimalloc backend on/off (the dependency + the TF_WITH_MIMALLOC define
    # together). Off ships the operator new/delete backend, no mimalloc dependency.
    options = {"with_mimalloc": [True, False]}
    default_options = {"with_mimalloc": True}

    def export_sources(self):
        repo_root = os.path.join(self.recipe_folder, "..", "..")
        copy(self, "CMakeLists.txt", src=repo_root,
             dst=self.export_sources_folder)
        copy(self, "trueformConfig.cmake.in",
             src=repo_root, dst=self.export_sources_folder)
        copy(self, "*", src=os.path.join(repo_root, "cmake"),
             dst=os.path.join(self.export_sources_folder, "cmake"))
        copy(self, "*", src=os.path.join(repo_root, "include"),
             dst=os.path.join(self.export_sources_folder, "include"))
        copy(self, "*", src=os.path.join(repo_root, "licenses"),
             dst=os.path.join(self.export_sources_folder, "licenses"))
        copy(self, "LICENSE*", src=repo_root, dst=self.export_sources_folder)

    def set_version(self):
        content = load(self, os.path.join(
            self.recipe_folder, "..", "..", "CMakeLists.txt"))
        match = re.search(r'project\(trueform\s+VERSION\s+([0-9.]+)', content)
        if match:
            self.version = match.group(1)

    def configure(self):
        self.options["onetbb"].tbbbind = False
        if self.options.with_mimalloc:
            # mimalloc must be MI_OVERRIDE=OFF (global override crashes when
            # static-linked) and static (we bake it into the consumer, not ship a dll).
            self.options["mimalloc"].override = False
            self.options["mimalloc"].shared = False

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("onetbb/[>=2020.0]",
                      transitive_headers=True, transitive_libs=True)
        if self.options.with_mimalloc:
            self.requires("mimalloc/3.3.2",
                          transitive_headers=True, transitive_libs=True)

    def validate(self):
        if self.settings.compiler.cppstd:
            try:
                if int(str(self.settings.compiler.cppstd)) < 17:
                    raise ConanInvalidConfiguration(
                        "trueform requires at least C++17")
            except ValueError:
                pass

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["TF_BUILD_PYTHON"] = "OFF"
        tc.variables["TF_BUILD_BENCHMARKS"] = "OFF"
        tc.variables["TF_BUILD_EXAMPLES"] = "OFF"
        tc.variables["TF_BUILD_TESTS"] = "OFF"
        tc.variables["TF_BUILD_VTK_INTEGRATION"] = "OFF"
        tc.variables["TF_USE_MIMALLOC"] = bool(self.options.with_mimalloc)
        if self.options.with_mimalloc:
            # use the Conan-provided mimalloc instead of fetching/vendoring our own
            tc.variables["TF_USE_SYSTEM_MIMALLOC"] = "ON"
        tc.generate()

        deps = CMakeDeps(self)
        if self.options.with_mimalloc:
            # our CMake links the target name `mimalloc-static`
            deps.set_property("mimalloc", "cmake_target_name", "mimalloc-static")
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        # Only trueform's own licenses. Dependency licenses (TBB, mimalloc) ship
        # with their own Conan packages, so a "LICENSE*" sweep would wrongly bundle
        # e.g. LICENSE.mimalloc even though we never vendor mimalloc here.
        licenses_dst = os.path.join(self.package_folder, "licenses")
        for lic in ("LICENSE", "LICENSE.noncommercial"):
            copy(self, lic, src=self.source_folder, dst=licenses_dst,
                 keep_path=False)

    def package_id(self):
        # Header-only: independent of compiler/build_type, but with_mimalloc
        # changes the propagated dependency + define, so keep it in the id.
        self.info.settings.clear()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "trueform")
        self.cpp_info.set_property("cmake_target_name", "tf::trueform")
        reqs = ["onetbb::onetbb"]
        if self.options.with_mimalloc:
            reqs.append("mimalloc::mimalloc")
            self.cpp_info.defines = ["TF_WITH_MIMALLOC"]
        self.cpp_info.requires = reqs
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []

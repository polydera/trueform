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
    description = "Real-time geometric processing. Easy to use, robust on real-world data."
    topics = ("geometry", "mesh", "computational-geometry", "mesh-processing",
              "mesh-boolean", "collision-detection", "csg", "header-only")

    settings = "os", "arch", "compiler", "build_type"
    package_type = "header-library"

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

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("onetbb/[>=2020.0]",
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
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(self, "LICENSE*", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"), keep_path=False)

    def package_id(self):
        self.info.clear()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "trueform")
        self.cpp_info.set_property("cmake_target_name", "tf::trueform")
        self.cpp_info.requires = ["onetbb::onetbb"]
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []

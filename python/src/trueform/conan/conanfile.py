"""
Conan recipe for trueform (header-only) - pip-installed package version.

This conanfile is designed to work with pip-installed trueform package.
For git repo usage, use: conan create conan/trueform

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import os
from conan import ConanFile
from conan.tools.files import copy
from conan.errors import ConanInvalidConfiguration, ConanException


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
    no_copy_source = True

    def _get_package_root(self):
        """Get trueform package root (parent of conan module)."""
        return os.path.join(self.recipe_folder, "..")

    def set_version(self):
        # Read version from _version.py (can't import - conan runs outside venv)
        import re
        version_file = os.path.join(self.recipe_folder, "..", "_version.py")
        if os.path.exists(version_file):
            with open(version_file) as f:
                content = f.read()
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                self.version = match.group(1)
                return
        raise ConanException(
            "Could not determine trueform version. Is the package installed correctly?")

    def export_sources(self):
        root = self._get_package_root()
        copy(self, "*", src=os.path.join(root, "include"),
             dst=os.path.join(self.export_sources_folder, "include"))

    def requirements(self):
        self.requires("onetbb/[>=2020.0]",
                      transitive_headers=True, transitive_libs=True)

    def configure(self):
        self.options["onetbb"].tbbbind = False

    def validate(self):
        if self.settings.compiler.cppstd:
            try:
                if int(str(self.settings.compiler.cppstd)) < 17:
                    raise ConanInvalidConfiguration(
                        "trueform requires at least C++17")
            except ValueError:
                pass

    def build(self):
        pass  # Header-only, nothing to build

    def package(self):
        copy(self, "*", src=os.path.join(self.source_folder, "include"),
             dst=os.path.join(self.package_folder, "include"))

    def package_id(self):
        self.info.clear()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "trueform")
        self.cpp_info.set_property("cmake_target_name", "tf::trueform")
        self.cpp_info.requires = ["onetbb::onetbb"]
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []

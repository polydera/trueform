#!/usr/bin/env python3
"""Refuse internal process files in the tracked tree.

Working documents — plans, task briefs, handoffs, reviews, analyses,
research notes — live outside the tree; git history and task records
own process. This gate holds the law mechanically, as a closed set
rather than a pattern chase: the repository root names exactly the
files and directories it carries, an underscore-caps document is a
working document wherever it stands, and a PDF is never tracked. It
walks `git ls-files` and fails on any violation, so a stray `git add`
cannot survive CI. Run from the repo root; exit 1 on any hit.
"""
import re
import subprocess
import sys

ROOT_FILES = {
    ".clang-format", ".gitignore", "AGENTS.md", "CLAUDE.md",
    "CMakeLists.txt", "COMMERCIAL.md", "CONTRIBUTING.md", "LICENSE",
    "LICENSE.noncommercial", "README.md", "RELEASE_NOTES.md",
    "pyproject.toml", "trueformConfig.cmake.in",
}
ROOT_DIRS = {
    ".claude", ".github", "agents", "benchmarks", "cmake", "conan",
    "cpp", "docs", "examples", "include", "licenses", "nuget",
    "python", "research", "tests", "typescript", "verify", "vtk",
}
CAPS_DOC = re.compile(r"(^|/)[A-Z][A-Z0-9]*(_[A-Z0-9]+)+\.(md|txt)$")
CAPS_DOC_ALLOWED = {"RELEASE_NOTES.md"}
FORBIDDEN_DIRS = re.compile(
    r"^(docs/superpowers/|experimentation/|local_tests/|\.codex/|"
    r"\.claude/worktrees/|scratchpad)")


def violation(path):
    if "/" not in path:
        if path not in ROOT_FILES:
            return "root file outside the closed set"
        return None
    if path.split("/", 1)[0] not in ROOT_DIRS:
        return "top-level directory outside the closed set"
    if FORBIDDEN_DIRS.match(path):
        return "internal directory"
    if CAPS_DOC.search(path) and path not in CAPS_DOC_ALLOWED:
        return "underscore-caps working document"
    if path.endswith(".pdf"):
        return "tracked PDF"
    return None


files = subprocess.run(["git", "ls-files", "-z"], capture_output=True,
                       text=True, check=True).stdout.split("\0")
files = [f for f in files if f]
bad = [(f, why) for f in files if (why := violation(f))]
for f, why in bad:
    print(f"TRACKED INTERNAL FILE: {f} ({why})")
print("tree hygiene:", "FAILED" if bad else f"clean ({len(files)} tracked)")
sys.exit(1 if bad else 0)

#!/usr/bin/env python3
"""Scan for MSVC portability violations (engineering rules, section 10).

Clang accepts all of these silently; they only fail on Windows. Run
from the repo root before landing any C++ change:

    python3 python/tools/portability_scan.py

Checks:
  10.1  local constexpr variable referenced inside a lambda body -- ANY
        reference, including if-constexpr conditions. MSVC 19.3x fails
        even non-odr-use constant folds of enclosing-scope constexpr
        variables from lambda bodies in templates ("read of a variable
        outside its lifetime", proven 2026-07 by tangential_relaxation).
        Fix: declare the constexpr inside the lambda body.
  10.3  non-ASCII characters in Catch2 TEST_CASE/SECTION names
  M_*   POSIX math macros (M_PI etc.) -- not standard C++, absent on
        MSVC without _USE_MATH_DEFINES. Use tf::pi<T> / tf::two_pi<T>.
  intrinsics: __int128 / __uint128 / __builtin_* outside tf::exact
        and vendored external code

10.2 (structured-binding default captures) needs real scope analysis to
avoid drowning in same-scope false positives; until then it stays a
review-time rule.

Pre-existing Windows-green hits live in portability_baseline.txt and the
scan is a ratchet: it fails only on NEW findings. Refresh the baseline
with --update-baseline after deliberately fixing old entries.

Exit code 1 on any non-baselined violation.
"""
import glob
import os
import re
import sys

bad = []

MATH_MACROS = re.compile(
    r"\b(M_PI|M_PI_2|M_PI_4|M_1_PI|M_2_PI|M_E|M_LOG2E|M_LOG10E|M_LN2|"
    r"M_LN10|M_SQRT2|M_SQRT1_2|M_2_SQRTPI)\b")

sources = [f for f in
           glob.glob("include/trueform/**/*.hpp", recursive=True) +
           glob.glob("tests/**/*.cpp", recursive=True) +
           glob.glob("tests/**/*.hpp", recursive=True) +
           glob.glob("examples/**/*.cpp", recursive=True) +
           glob.glob("python/src/**/*.cpp", recursive=True) +
           glob.glob("python/include/**/*.hpp", recursive=True) +
           glob.glob("vtk/**/*.cpp", recursive=True) +
           glob.glob("vtk/**/*.hpp", recursive=True)
           if "/external/" not in f]

for f in sources:
    s = open(f, encoding="utf-8").read()
    if f.startswith("include/") and "/exact/" not in f:
        for m in re.finditer(r"__int128|__uint128|__builtin_\w+", s):
            bad.append(f"{f}: intrinsic '{m.group(0)}' outside tf::exact")
            break
    for m in MATH_MACROS.finditer(s):
        line = s.count("\n", 0, m.start()) + 1
        bad.append(f"{f}:{line} M_*: '{m.group(0)}' -- use tf::pi<T> "
                   "(not defined on MSVC)")
    lines = s.split("\n")
    for i, ln in enumerate(lines):
        m = re.match(r"\s+constexpr\s+[\w:<>]+\s+(\w+)\s*=", ln)
        if not m or re.match(r"\s*static\b", ln):
            continue
        name = m.group(1)
        # window: to the end of the enclosing (namespace-level) function —
        # the next closing brace at column 0 — not a fixed line count
        end = min(i + 400, len(lines))
        for j in range(i + 1, min(i + 400, len(lines))):
            if lines[j].startswith("}"):
                end = j
                break
        window = "\n".join(lines[i + 1:end])
        # negative lookbehind: subscripts follow an identifier, ')' or
        # ']'; capture lists do not
        for lm in re.finditer(r"(?<![\w\)\]&])\[[^\]\n]*\][^{;]*\{", window):
            start = lm.end()
            depth, j = 1, start
            while depth and j < len(window):
                depth += (window[j] == "{") - (window[j] == "}")
                j += 1
            body = window[start:j]
            # the documented fix is re-declaring the constexpr inside the
            # lambda; a shadowing declaration clears the reference
            if re.search(r"constexpr[^\n=]*\b" + name + r"\s*=", body):
                continue
            if re.search(r"\b" + name + r"\b", body):
                bad.append(f"{f}:{i + 1} 10.1: local constexpr '{name}' "
                           "referenced in lambda")
                break

for f in glob.glob("tests/**/*.cpp", recursive=True):
    for i, ln in enumerate(open(f, encoding="utf-8"), 1):
        if re.search(r"TEST_CASE|SECTION|TEMPLATE_TEST_CASE", ln):
            if any(ord(c) > 127 for c in ln):
                bad.append(f"{f}:{i} 10.3: non-ASCII in test name")

for f in sources:
    for i, ln in enumerate(open(f, encoding="utf-8"), 1):
        code = ln.split("//")[0]
        if re.search(r"(?:^|[\s(,])(?:near|far)\s*[=(]", code):
            bad.append(f"{f}:{i} 10.4: 'near'/'far' identifier "
                       "(windows.h macro)")

baseline_path = os.path.join(os.path.dirname(__file__),
                             "portability_baseline.txt")
if "--update-baseline" in sys.argv:
    open(baseline_path, "w").write("\n".join(sorted(bad)) + "\n")
    print(f"baseline updated: {len(bad)} entries")
    sys.exit(0)
baseline = set()
if os.path.exists(baseline_path):
    baseline = {ln.strip() for ln in open(baseline_path) if ln.strip()}
new = [b for b in bad if b not in baseline]
for b in new:
    print("VIOLATION:", b)
stale = baseline - set(bad)
if stale:
    print(f"note: {len(stale)} baseline entries no longer fire")
print("portability scan:", "FAILED" if new else
      f"clean ({len(bad)} baselined)")
sys.exit(1 if new else 0)

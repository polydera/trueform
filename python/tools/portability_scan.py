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
  10.4  'near'/'far' identifiers -- empty macros from minwindef.h in
        any TU a Windows header reaches (NOMINMAX does not cover them)
  10.5  a default-capture ([&]/[=]) generic lambda decomposing one of
        its own parameters (the zip shape) -- MSVC loses the enclosing
        scope after the structured binding and every implicitly
        captured local goes undeclared, by value as much as by
        reference. List what the body reads in the capture list;
        capture-less lambdas are fine too
  10.6  a lambda with an explicit capture list reading a const local of
        the enclosing function that the list does not name -- Clang and
        GCC read a const integral initialised from a constant without
        capturing it, MSVC refuses the implicit capture (C3493). List it
  intrinsics: __int128 / __uint128 / __builtin_* outside tf::exact
        and vendored external code
  10.7  a reference bound to a member of a site a decision view hands
        out -- `const auto &a = sites[k].pt;` -- where the aligned
        decision's operator[] returns BY VALUE: Clang and GCC keep the
        temporary alive for the reference, MSVC frees it and the
        reference reads zeros (proven 2026-11 by the deep lattice
        search walking the whole lattice on Windows). Copy the member.
  draws: a std distribution (std::*_distribution) or std::shuffle in
        tests/ -- the engines are standardized, the distributions and
        the shuffle are implementation-defined, so a seeded test draws
        another input on MSVC's STL. Draw through
        tests/common/uniform_draw.hpp

  7.1   inside a GENERIC lambda (a parameter declared 'auto'), a read of
        a local or parameter of the enclosing function whose name is a
        namespace-scope tf template under include/trueform/ (frame,
        points, tree, buffer, range, faces...) -- MSVC's template-time
        lookup finds the template (C2955). Name frame_of results 'pose'.

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

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

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

STD_DRAW = re.compile(r"\bstd::(?:\w+_distribution|shuffle)\b")
for f in sources:
    if not f.startswith("tests/"):
        continue
    for i, ln in enumerate(open(f, encoding="utf-8"), 1):
        m = STD_DRAW.search(ln.split("//")[0])
        if m:
            bad.append(f"{f}:{i} draws: '{m.group(0)}' is implementation-"
                       "defined -- draw through tests/common/uniform_draw.hpp")

SITE_MEMBER_REF = re.compile(r"\bauto\s*&&?\s*\w+\s*=\s*sites\[.*?\]\.\w+\s*;")
for f in sources:
    for i, ln in enumerate(open(f, encoding="utf-8"), 1):
        m = SITE_MEMBER_REF.search(ln.split("//")[0])
        if m:
            bad.append(f"{f}:{i} 10.7: '{m.group(0).strip()}' binds a reference "
                       "to a member of a site a decision hands out by value "
                       "-- MSVC frees the temporary; copy the member")

TEST_NAME_MACRO = re.compile(
    r"\b(?:TEST_CASE|TEMPLATE_TEST_CASE|TEMPLATE_TEST_CASE_SIG|SECTION|"
    r"SCENARIO|GIVEN|WHEN|THEN)\s*\(")
STRING_LITERAL = re.compile(r'"(?:[^"\\\n]|\\.)*"')

for f in glob.glob("tests/**/*.cpp", recursive=True):
    s = open(f, encoding="utf-8").read()
    for m in TEST_NAME_MACRO.finditer(s):
        i, depth = m.end(), 1
        while i < len(s) and depth:
            depth += (s[i] == "(") - (s[i] == ")")
            i += 1
        # the registered name is the macro's literals, which a wrapped
        # invocation spreads over several lines
        if any(ord(c) > 127 for lit in STRING_LITERAL.finditer(s[m.end():i - 1])
               for c in lit.group(0)):
            line = s.count("\n", 0, m.start()) + 1
            bad.append(f"{f}:{line} 10.3: non-ASCII in test name")

for f in sources:
    for i, ln in enumerate(open(f, encoding="utf-8"), 1):
        code = ln.split("//")[0]
        if re.search(r"(?:^|[\s(,])(?:near|far)\s*[=(\[;,)]", code):
            bad.append(f"{f}:{i} 10.4: 'near'/'far' identifier "
                       "(windows.h macro)")

LAMBDA_INTRO = re.compile(r"(?<![\w\)\]&])(\[[^\]\n]*\])\s*\(")
DEFAULT_CAPTURE = re.compile(r"^\[\s*[&=]\s*(?:\]|,)")
GENERIC_PARAM = re.compile(r"\bauto\b\s*&{0,2}\s*(\w+)")
DECOMPOSED = re.compile(r"\bauto\b\s*&{0,2}\s*\[[^\]\n]*\]\s*=\s*(\w+)\b")
for f in sources:
    file_lines = open(f, encoding="utf-8").read().split("\n")
    for i, ln in enumerate(file_lines):
        for m in LAMBDA_INTRO.finditer(ln):
            if not DEFAULT_CAPTURE.match(m.group(1)):
                continue
            # the window opens on the introducer's line, so m's offsets
            # into that line are offsets into the window
            window = "\n".join(file_lines[i:i + 7])
            j, depth = m.end(), 1
            while j < len(window) and depth:
                depth += (window[j] == "(") - (window[j] == ")")
                j += 1
            names = set(GENERIC_PARAM.findall(window[m.end():j - 1]))
            hit = next((d.group(1) for d in DECOMPOSED.finditer(window[j:])
                        if d.group(1) in names), None)
            if hit:
                bad.append(f"{f}:{i + 1} 10.5: default-capture generic lambda "
                           f"decomposes '{hit}' -- MSVC loses the enclosing "
                           "scope after the binding; list what the body reads "
                           "in the capture list")

# only a const initialised from a constant (a qualified name or a literal)
# is read without capture by Clang and GCC; one initialised at run time
# fails on every compiler, so it is never a Windows-only finding
CONST_LOCAL = re.compile(r"\s+const\s+[\w:<>]+\s+(\w+)\s*=\s*(?:[A-Za-z_][\w:]*|[\d.]+[uUlLfF]*)\s*;")
EXPLICIT_CAPTURE = re.compile(r"^\[(?!\s*[&=]\s*(?:\]|,))[^\]]*\]$")
for f in sources:
    lines = open(f, encoding="utf-8").read().split("\n")
    for i, ln in enumerate(lines):
        m = CONST_LOCAL.match(ln)
        if not m or re.match(r"\s*static\b", ln) or "constexpr" in ln:
            continue
        name = m.group(1)
        end = min(i + 400, len(lines))
        for j in range(i + 1, end):
            if lines[j].startswith("}"):
                end = j
                break
        # comments and string literals are not code; a structured
        # binding's bracket list is not a capture list
        window = "\n".join(STRING_LITERAL.sub('""', ln.split("//")[0])
                           for ln in lines[i + 1:end])
        for lm in re.finditer(r"(?<![\w\)\]&])(\[[^\]\n]*\])\s*(\([^)]*\))?[^{;]*\{", window):
            captures = lm.group(1)
            if not EXPLICIT_CAPTURE.match(captures):
                continue
            if re.search(r"\bauto\s*&{0,2}\s*$", window[:lm.start()].rsplit("\n", 1)[-1]):
                continue
            if re.search(r"\b" + name + r"\b", captures):
                continue
            if lm.group(2) and re.search(r"\b" + name + r"\b", lm.group(2)):
                continue
            start = lm.end()
            depth, j = 1, start
            while depth and j < len(window):
                depth += (window[j] == "{") - (window[j] == "}")
                j += 1
            body = window[start:j]
            # a declaration, a range-for variable or a structured binding
            # of the same name inside the body shadows the local
            if (re.search(r"[\w:<>]\s*&{0,2}\s*\b" + name + r"\s*[=;:(),]", body)
                    or re.search(r"\[[^\]]*\b" + name + r"\b[^\]]*\]\s*=", body)):
                continue
            if re.search(r"(?<![\.\w>:])" + name + r"\b", body):
                line = i + 1 + window[:lm.start()].count("\n") + 1
                bad.append(f"{f}:{line} 10.6: const local '{name}' read in a "
                           "lambda that does not list it -- MSVC refuses the "
                           "implicit capture (C3493); list it")
                break

# 7.1: the names of the namespace-scope tf templates (column 0, as the
# house format does not indent namespaces)
TF_TEMPLATE = re.compile(
    r"^template\s*<[^;{}]*?>\s*(?:struct|class|using)\s+(\w+)", re.M)
tf_templates = set()
for f in glob.glob("include/trueform/**/*.hpp", recursive=True):
    if "/external/" not in f:
        tf_templates.update(TF_TEMPLATE.findall(open(f, encoding="utf-8").read()))
DECLARED = re.compile(r"\bauto\s*&{0,2}\s*(\w+)\s*(?:=|\{)")
GENERIC_LAMBDA = re.compile(
    r"(?<![\w\)\]&])(\[\s*[&=][^\]\n]*\])\s*\(([^)]*)\)[^{;]*\{")
SCOPE_ROOT = re.compile(r"^\s*(?:template\b.*)?(?:struct|class|namespace|union)\b")
CONTROL_HEAD = re.compile(r"^\s*(?:\}\s*)?(?:if|for|while|switch|else|do|try|catch)\b")
for f in sources:
    if not f.startswith("include/"):
        continue
    text = "\n".join(STRING_LITERAL.sub('""', ln.split("//")[0])
                     for ln in open(f, encoding="utf-8").read().split("\n"))
    seen = set()
    for lm in GENERIC_LAMBDA.finditer(text):
        if not re.search(r"\bauto\b", lm.group(2)):
            continue
        depth, j = 1, lm.end()
        while depth and j < len(text):
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        body = text[lm.end():j]
        params, captures = lm.group(2), lm.group(1)
        # the enclosing function's text before the lambda: walk back
        # through the scopes still open at the lambda, up to the first
        # struct/namespace opener
        before, k, d, enclosing = text[:lm.start()], lm.start(), 0, []
        pos = k
        while pos > 0:
            pos -= 1
            c = before[pos]
            if c == "}":
                d += 1
            elif c == "{":
                if d:
                    d -= 1
                    continue
                head = before[before.rfind("\n", 0, pos) + 1:pos + 1]
                if SCOPE_ROOT.match(head):
                    break
                enclosing.append((pos, k))
                k = pos
        if not enclosing:
            continue
        # open-scope texts: declarations at depth 0 of each open scope,
        # plus the function head's parameters
        locals_ = set()
        open_at = [e[0] for e in enclosing]
        top = open_at[-1]
        head_start = before.rfind(";", 0, top) + 1
        head_start = max(head_start, before.rfind("}", 0, top) + 1)
        scope_text = before[head_start:lm.start()]
        # a scope closed before the lambda declares nothing the lambda sees
        frames = [[]]
        for c in scope_text:
            if c == "{":
                frames.append([])
            elif c == "}":
                if len(frames) > 1:
                    frames.pop()
            else:
                frames[-1].append(c)
        flat = "".join("".join(fr) + "\n" for fr in frames)
        locals_ = tf_templates.intersection(DECLARED.findall(flat))
        for name in locals_:
            if (re.search(r"[\w>&*]\s+&{0,2}" + name + r"\s*(?:=|;|\{|,|\)|:)", body)
                    or re.search(r"\b" + name + r"\b", captures)
                    or re.search(r"\b" + name + r"\b", params)):
                continue
            rd = re.search(r"(?<![\.\w>:])" + name + r"\b(?!\s*::)", body)
            if rd:
                line = text.count("\n", 0, lm.end() + rd.start()) + 1
                if (line, name) not in seen:
                    seen.add((line, name))
                    bad.append(f"{f}:{line} 7.1: local '{name}' read in a "
                               "generic lambda is also a tf template -- "
                               "MSVC template-time lookup finds the "
                               "template (C2955); rename the local")

baseline_path = os.path.join(os.path.dirname(__file__),
                             "portability_baseline.txt")
if "--update-baseline" in sys.argv:
    notes = []
    if os.path.exists(baseline_path):
        notes = [ln.rstrip() for ln in open(baseline_path) if ln.startswith("#")]
    open(baseline_path, "w").write("\n".join(notes + sorted(bad)) + "\n")
    print(f"baseline updated: {len(bad)} entries")
    sys.exit(0)
baseline = set()
if os.path.exists(baseline_path):
    baseline = {ln.strip() for ln in open(baseline_path)
                if ln.strip() and not ln.startswith("#")}
new = [b for b in bad if b not in baseline]
for b in new:
    print("VIOLATION:", b)
stale = baseline - set(bad)
if stale:
    print(f"note: {len(stale)} baseline entries no longer fire")
print("portability scan:", "FAILED" if new else
      f"clean ({len(bad)} baselined)")
sys.exit(1 if new else 0)

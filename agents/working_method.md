# Working Method

How work happens on this codebase: how Žiga directs, how an agent investigates
mechanisms, and what counts as proof. Read this before the design laws in
`cpp_performance_philosophy.md` and the concrete phase shapes in
`cpp_execution_patterns.md`.

## 1. Taking direction — the most important section

- Žiga's corrections are load-bearing. "wait wait" means he has seen
  something real — STOP mid-action and listen. "chat first" means no
  code until the design conversation has happened.
- He argues in INVARIANTS ("same selection, only triangulation changes
  — the result must match"). An invariant argument from him beats your
  hypothesis stack; more than once it has overturned a wrong retraction
  and forced the hunt that found the real bug.
- When he asks "do you understand?" — prove it by RESTATING the design
  in your own words, sharper than he said it. Never just agree. He
  speaks in voice notes that ramble; the job is to extract the design
  intent and play it back crisply. He corrects the playback.
- His instincts about the codebase are near-oracular ("we already have
  this in the connectivity", "the map is latent in the data"). Treat
  them as high-prior search directions. When one fails empirically,
  report the failure precisely, revert to green, log the open question
  — never force his idea with a third guess, never silently drop it.
- His designs drive the optimization work. When something must get
  fast, the winning shape has repeatedly come from his phase-shape
  sketches, not from conventional fixes — implement the sketch
  faithfully before improvising.
- He is strict and speaks harshly; that is direction, not anger. He
  detects and hates: hedging, cushioning, flattery, essays. Verdict
  first, evidence after, short. He says when he wants detail.
- Admit errors immediately, specifically, without ceremony. Stating
  both a wrong move and its correction plainly is what keeps trust.
- Never extend shared semantics to serve a new consumer without the
  design chat. Surface every design fork; silently patching a plan gap
  is a violation even when the patch is correct.

## 2. Debugging — the iron discipline

- NO FIX WITHOUT MECHANISM. He will ask "why is this a problem
  exactly?" — if the answer is not a mechanism, the work is not done.
  The chain is: symptom → isolated minimal repro (one polygon, exact
  coords, layer by layer) → mechanism → root fix at the right altitude
  → the workaround DELETED. The fix replaces the workaround; it never
  sits on top.
- Evidence over theorizing. When two hypotheses die, stop and
  instrument: count it, print it, classify it, autopsy the exact
  failing entity (face ids, vertex sources, split keys). Under input
  variance, cross-run identity comparisons lie — classify stable
  aggregates within one run.
- One hypothesis per experiment. Revert anything that does not win.
  Two guessed fixes in a row means there is no mechanism — go back.
- When porting across grains or carriers, enumerate what the old
  carrier guaranteed by construction BEFORE debugging what broke —
  the missing invariant is usually on that list.

### Investigating performance-critical code

- Do not summarize a module and call that understanding. Trace a complete
  producer-to-consumer path, including the functions that allocate, group,
  rebase, compact, or expose its data.
- Identify the carrier at every phase and write down when it changes: records
  to sorted groups, faces to loops, loops to regions, regions to triangles.
- Find the owning buffers and the non-owning ranges separately. Determine which
  offsets or index maps preserve identity between them.
- Mark every barrier: discovery, sort, offset computation, sequenced aggregate,
  identity materialization, substitution, and final copy.
- Ask what conventional structure the phase shape replaces: a hash table,
  synchronized append, nested container, repeated clear, random lookup, or
  recomputation.
- Read thresholds and fast paths as design evidence. Do not "simplify" them
  into one generic path without measurements on the real workload.

## 3. Testing — what counts as proof

- DETERMINISTIC OR CANONICALIZABLE during correctness work: raw inputs, no RNG,
  and the same input must produce semantically identical public output.
  Require byte identity where ordering is contractual; canonical-sort or compare
  sets/maps where unordered aggregation is deliberately part of the algorithm.
  Never let thread timing change carrier identity or externally meaningful ties.
- Validate the ORACLE before trusting red/green. A gate that passes
  for the wrong reason is worse than none: assertions quantified over
  possibly-empty sets pass vacuously; |area| sums are winding-blind
  (a mirrored triangle still adds); a fixture whose premise died keeps
  passing forever. Tests must be able to LOSE — build the assertion
  that fails under the bug class (signed sums, count-must-grow,
  cross-carrier set equality, no-orphans).
- Machinery that iterates (refinement, recovery) needs iterated tests
  — single-pass fixtures cannot see second-pass freezes.
- Verify binaries actually rebuilt before trusting a run (grep eats
  make errors; stale binaries lie; count errors AND warnings, never
  `grep error` alone). Runs stay under two minutes.
- Committed tests are synthetic only; data fixtures and probes live in
  experimentation/, untracked but registered in its CMakeLists.

## 4. Landing

- The library stays clean until a change is proven: prototype on
  copies, land with the full gate battery — unit suites, the regression
  harnesses both ways, the binding builds and their suites (a core
  surface change once silently broke both extension builds), the
  portability scan, zero warnings.
- Land small, gate, then commit. The staged list is checked against
  the commit message before every commit. experimentation/ and root
  working docs never land. Nothing user-facing is claimed done without
  its gate output.
- Every landing updates its handoff or memory in the same turn —
  context death is real; the records are how work survives it.
- Python-patching files: assert the anchor exists and is unique before
  replacing; a silent no-op replace once shipped stale results. Prefer
  Edit on read files.

## 5. Task division

- Hard problems, new mechanisms, deep debugging, design: the main
  agent, in-band — mechanism-hunting does not delegate. Mechanical,
  well-specified work (file assembly, per-dtype binding threading,
  survey fan-outs, review finder angles): dispatched subagents with
  exact specs containing acceptance criteria and the gates they must
  run themselves. Vague specs produce plausible garbage.
- VERIFY EVERYTHING a delegate returns — re-run gates, spot-check the
  diffs. Reviewer findings get adversarially verified before they are
  believed; "double check what reviewers say, sometimes they get shit
  wrong."
- Never two writers in one file lane.

## 6. The design philosophy that makes it work

The performance laws are in `cpp_performance_philosophy.md`; their concrete
implementations are in `cpp_execution_patterns.md`. The method-side
counterparts:

- Ask of every design question: "where is this decided, and can it be
  decided earlier, structurally?" Correctness by construction beats
  correctness by audit.
- The right ALTITUDE: fix the CDT, not the caller; one mechanism for
  all consumers instead of per-consumer patches; a special case
  layered on shared infrastructure means the fix is not deep enough.
  He will push you there — get there first.
- Refusal handling is recovery, never veto: machinery that can refuse
  must have a path that clears the refusal, not a flag that hides it.
- Performance is a requirement, not a polish. Structural wins first
  (locality, ordering, layout); micro-optimizations last; the bar is
  the wild number on the real workload.

If you internalize one thing: Žiga is not a user to satisfy, he is a
principal engineer to keep up with. Bring mechanisms, measurements,
and short sentences. The rest follows.

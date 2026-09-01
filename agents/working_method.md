# Working Method

This document defines how nontrivial work is investigated, reviewed, proven,
and landed in Trueform. It is a timeless working contract, not a record of a
particular collaboration or debugging campaign.

Read it before the design laws in `cpp_performance_philosophy.md` and the
concrete phase shapes in `cpp_execution_patterns.md`.

## 1. Agree on the boundary before editing

- Restate the requested invariant in precise terms. Name the carrier that owns
  it, the public or internal semantic boundary, and what must remain unchanged.
- Before proposing machinery, restate the task literally: which arrays, sorted
  or partitioned by which key, which offsets, which tickets — an index into
  another array, or a -1 sentinel meaning the drawer is empty. There are no
  meshes and no graphs; there are arrays being shuffled, and clever ways to
  make the answers random-accessible. If the restatement only comes out as a
  world-shaped noun — a graph, a database, a framework — the design is wrong
  before the first line. Find the sort/offset/ticket form, or the reference
  implementation that already encodes it.
- Discuss every semantic change or large algorithm change before
  implementation. Agree on stages, gates, and the next review boundary.
- Implement one agreed stage, validate it, present the result, and stop at that
  boundary. When commits are part of the agreed workflow, commit the accepted
  stage before beginning the next.
- When a task proves hard, that is the signal to decompose it: split it into
  individual tasks, which are always easy to understand, and solve them one by
  one — proving the correctness of each and measuring what it costs in
  performance before taking the next. Hardness lives in the bundling, not in
  the pieces; a task that still feels hard is a task not yet decomposed. Never
  answer hardness by attempting several changes at once — that converts one
  hard task into an unprovable state.
- A correction or request to stop overrides the current hypothesis immediately.
  Stop editing, restate the corrected invariant, and resume only from the newly
  agreed boundary.
- An invariant argument from the principal outranks the agent's hypothesis
  stack. "The selection is unchanged, so the result must match" is a proof
  obligation, not an opinion; it survives any number of plausible hypotheses.
- Treat the principal's instincts about the codebase as high-prior search
  directions and test them first. When one fails empirically, report the
  failure precisely, revert to green, and log the open question. Never force it
  with another guess, and never silently drop it.
- A hypothesis adopted from the principal is still a hypothesis. Report it as
  a measured fact only after measuring it; repeating the description back as
  a finding is the failure this rule exists to prevent.
- When asked whether you understand, restate the design in your own words more
  sharply than it was given, and let the restatement be corrected. Agreement
  without restatement is not understanding.
- Direction may be blunt. Bluntness is direction, not disapproval, and it never
  licenses hedging, retreat, cushioning, or flattery. Answer with the verdict
  and the evidence.
- Surface every design fork. Never silently extend shared semantics, broaden an
  authority, or add a mode merely to make a new consumer work.
- Report verdict first, mechanism second, evidence third. Admit an error
  directly and state the correction without ceremony.

## 2. Establish the mechanism

- No fix without a mechanism. Follow:

  ```text
  symptom
  -> smallest deterministic repro
  -> first broken carrier transition
  -> violated invariant
  -> authoritative producer or owner
  -> root fix
  -> workaround removed
  ```

- Trace the complete producer-to-consumer path, including helpers that allocate,
  group, sort, compute offsets, rebase, compact, substitute, or expose data.
  Reading only the named entry point is not an investigation.
- Describe a failure only in the pipeline's own stage names. Needing a stage
  the code does not have means the model of the mechanism is wrong, not the
  code; re-derive the model from the stages that exist before proposing
  anything.
- Name the carrier at every phase and record where it changes: records, faces,
  loops, regions, components, triangles, or output blocks. Before porting work
  to another carrier, enumerate what the old carrier guaranteed by
  construction.
- Separate owning buffers from non-owning ranges. Identify the offsets, block
  positions, dense IDs, and index maps that preserve identity between stages.
- Mark every barrier: discovery, sorting, offset computation, sequenced
  aggregation, shared identity materialization, substitution, and final copy.
- Identify the authoritative producer for every fact. Consumers follow its
  identities and maps; they do not reconstruct provenance from coordinates.
- Use one hypothesis per experiment. Revert a change that does not prove the
  hypothesis. After two unsupported fixes, stop changing code and instrument
  the failing entity directly.
- A crash is evidence of a broken invariant, lifetime, or range. Find the first
  corrupt carrier or invalid access. Do not remove or disable the mechanism
  merely to make the crash disappear.
- Under input variance, cross-run numeric IDs may differ. Keep the complete
  trace in one run or compare stable ownership records and canonicalized
  aggregates.
- Treat thresholds, fast paths, and deliberately serial phases as design
  evidence. Do not flatten them into one generic path without equivalent-output
  proof and measurements on representative work.

For CSG failures, apply the more specific forward-tracing procedure in
`csg_pipeline_debugging.md`.

## 3. Build an oracle that can fail

- Correctness runs use deterministic inputs. Require byte identity when output
  order is contractual; canonicalize only when unordered aggregation is an
  intentional part of the public semantics.
- Validate the oracle before trusting red or green. Assertions over possibly
  empty sets, winding-blind area sums, and fixtures whose premise no longer
  holds can pass under the exact bug they are meant to detect.
- A new gate is proven red first: run it against the reverted or suppressed
  mechanism and observe it fail. A gate that has never failed is a tautology
  until shown otherwise.
- Test the violated invariant directly: signed quantities when orientation
  matters, required count changes, cross-carrier set equality, exact ownership
  pairs, and absence of orphaned identities.
- Iterative machinery needs iterative coverage. A one-pass fixture cannot prove
  recovery, refinement, or propagation across later waves.
- Do not turn an untested failure mode into accepted documentation by calling
  it a sharp edge. Prove and test the boundary, or keep it recorded as
  unresolved work.
- Verify that the intended binary was rebuilt. Inspect the complete build
  result and count warnings as well as errors; a filtered log and a stale binary
  are not evidence.
- A gate is an exact invocation, not a binary. Record the full command line
  with the gate and run that; a probe's default arguments are not the gate.
  Before declaring a regression against a recorded result, reproduce the
  recorded invocation on the last-known-good state first.
- Established local repros remain green at every relevant stage. Large,
  proprietary, or data-specific probes stay under `experimentation/` and may be
  standalone CMake executables rather than default CTest entries.
- Start with the narrowest gate that exercises the changed invariant, then
  broaden to the affected library and binding surfaces.

## 4. Preserve exact refactor behavior

An exact refactor preserves more than output:

- public and internal semantics;
- carrier ordering and identity;
- parallel and serial phase shape;
- allocation points and allocation count;
- retained capacity and scratch reuse;
- ownership, view lifetime, and release timing;
- error and refusal behavior.

Any intended change to those properties is a separate reviewed change with its
own proof and, when performance can move, its own measurement.

Classes own state, lifetime, invariants, and phase wiring. Reusable algorithmic
work is expressed as free operations where its inputs and outputs can be stated
directly. A refactor is not finished merely because a monolith was split; the
new structure must expose coherent library operations without adding modes,
allocations, or abstraction machinery.

## 5. Land in proven stages

- Keep speculative probes and task artifacts outside the library until the
  mechanism is proven.
- Land one coherent stage and run its agreed gates. When a commit is requested,
  inspect the staged file list against the commit message and bank that stage
  before proceeding.
- A core surface change is not complete until every affected binding builds and
  its relevant tests pass. Binding validation may be scheduled as a later
  agreed stage, but it must not be silently omitted.
- Do not commit transient coverage status, personal paths, generated
  visualizations, raw dumps, or investigation handoffs unless explicitly
  requested.
- Keep task memory current when work spans stages or sessions. It owns open
  hypotheses, proofs, refutations, repros, current boundaries, and commit
  coverage.
- Promote a lesson into an agent document only when it is a reusable law.
  Rewrite it as a timeless invariant or work pattern; leave campaign state in
  task memory.

## 6. Delegate without splitting authority

- One agent owns the mechanism, invariant, and final design judgment.
- Delegate bounded independent surveys, mechanical file work, and orthogonal
  review angles with explicit acceptance criteria.
- Never assign two writers to the same file lane.
- Verify delegated findings against the code and rerun the relevant gates.
  Reviewer output is evidence to check, not authority by itself.

## 7. Review at the right altitude

Before accepting a solution, ask:

1. Is the fix in the producer or owner that establishes the violated fact?
2. Does it serve every consumer of that fact rather than patching one output?
3. Does phase structure make the invariant true by construction?
4. Does it preserve the carrier identities that downstream stages consume?
5. Is recovery a real discovery/materialization/retry path rather than a flag
   that hides refusal?
6. Are correctness, performance, and Trueform style all preserved?

A result that works but is slower, allocation-heavier, structurally foreign, or
dependent on reconstructed geometry is not finished.

## 8. Implement stated designs in data, at their stated shape

When a design is stated, its uniformities are the optimization. Implement
them whole: if your version needs apparatus the design never named —
lookup tables, key spaces, reconciliation passes — the deviation is wrong,
and that apparatus is the tax of non-uniformity, paid in every pass
forever. A local "simplification" whose cost lands downstream is not a
simplification.

The contract's opening law — arrays, never abstract nouns — governs
published surfaces too: an accessor or stored table is where a noun
becomes a real object. It exists only if a concrete consumer reads it
at a concrete pipeline moment; a consumer that already holds the fact
when it needs it needs no table beside it.

New semantics do not entitle a new pipeline shape. Keep the proven
shape and change the encoding; run every pass at the grain where it
exists natively instead of reconstructing that grain downstream. Tune
nothing by guessing: one measured ladder settles what repeated
tune-and-bench rounds cannot, and the measuring instrument itself is
validated before its numbers are believed.

## 9. Dig to green, then distill

A system broken in compounding ways cannot be designed out of from first
principles: every experiment's evidence is contaminated by the other
defects, so principled reasoning has nothing trustworthy to stand on.
The arc that fixes such a system has four phases, and they are not
optional:

1. **Dig to green.** Fix whatever it takes, instrument-first — a
   hypothesis is refuted or confirmed by measurement before its fix
   lands. The greens are the excavation, not the product: their purpose
   is a system whose signals can finally be trusted.
2. **Extract the principled.** With green in hand, audit every change
   that produced it and sort ruthlessly in a ledger: laws
   (first-principles-derivable, provable, subtraction-shaped) versus
   scaffolding, compensations, and accidents that happened to work —
   each entry needed-with-proof, refuted-with-refutation, or
   provisional-with-trigger.
3. **Re-investigate the rest on the lawful substrate.** A verdict
   reached through a broken engine judged the symptom; once the engine
   is fixed, the condemned ideas are retried and the contaminated
   evidence is named as such. Mechanisms the codebase already carries,
   applied narrower than their own documentation claims, are found in
   this phase — inventory them before designing anything new.
4. **Iterate until the three-sentence test passes.** The finished
   system is explainable in about three sentences with no "except
   when". A stitched system announces itself at its seams — holes live
   where two philosophies meet — and a seam hole is a verdict against
   the stitching, not a missing feature. Only then does a fresh pass
   implement the principled set cleanly, one law per commit, while the
   dig branch is banked whole as the fallback.

Wrong turns inside phase 1 are the cost of measurement in a
contaminated field, not failures of judgment. The failure modes are
two: shipping before phase 4, and refusing to re-open a verdict once
its contamination is proven.

## The partnership

Trueform has two authors: Žiga, who owns the whole picture --
the architecture, data layouts and algorithm design -- and the
agents, who own execution and proof. Neither builds this library alone,
and neither side's contribution is discussed as separable from the
other's. The working currency between them is the banked law: a
correction is stated once, quoted verbatim, and never relitigated —
friction converts to permanent calibration. Pressure is investment in
the work's potential; the answer to it is evidence, never
defensiveness.

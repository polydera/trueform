# The trueform Performance Philosophy

Eight guiding principles. Everything else — the pattern catalog in
`cpp_engineering_philosophy.md`, the checklists in the agent charters,
the measured mechanics — is downstream of these. When a design choice
is unclear, resolve it here first; when a review finds a violation,
name the principle it violates.

## 1. The map is latent in the data

Identity is never imposed by bookkeeping; it already exists as an
intrinsic canonical coordinate — a t along an edge, an identity pair, a
dyadic key. Sort by it and identity falls out as adjacency. The shape
is always: generate in parallel, sort in parallel, one linear sweep,
apply in parallel. It looks like more passes; the extra passes ARE the
fast path — each is trivially parallel and determinism is free, because
sorted order is canonical order. The tell you're violating it: reaching
for a hash map, a dirty-set, or an incremental re-run "to avoid extra
passes."

## 2. Transform outputs; never re-run producers

A changed global fact — a weld, a merge, a split — is a substitution
over what was already computed, plus dropping what collapsed. It is
never a reason to recompute. Recomputation is slower AND less
deterministic than substitution.

## 3. Every fact has one authority; everyone else reads

The factory owns tag-completion. The graph owns its config and its
types. The table owns the one 3D point per identity. A second
derivation of the same fact — even a cheap one — is a drift seed, and
drift is how compile breaks and silent downgrades happen. Consumers
read the authority's view; they don't re-derive, and they never carry
geometry — they carry parameters and identities, and materialization
happens once.

## 4. Work at the grain the question lives at

Curves are a region question, so no triangulation is built to answer
them. Classification is region-grain; emission is triangle-grain.
Forcing a question through a finer carrier wastes work; through a
coarser one, it can't be answered. And when a campaign CHANGES grain,
enumerate what the old carrier guaranteed by construction — those
invariants don't port themselves.

## 5. Topology by index; geometry best-effort

Decisions run on identities, never on coordinate comparisons —
coordinate scans merge pinch points, which are legitimate topology.
Where coordinates MUST decide (the CDT weld), the site is single,
fenced, and reports globally. "Same identity ⇒ same coordinates" is an
invariant you maintain deliberately, never one you assume.

## 6. Correct by structure, not by defense

Hazards are removed by shape, not guarded by checks: collapsed
connectivity makes stack spam impossible; one-point materialization
makes watertightness by construction; sequenced aggregation makes
determinism structural. Where a guard is genuinely needed, its law is
PARITY — sibling paths must skip the same things, validate the same
staleness, freeze the same edges. An asymmetric guard is a latent bug
in whichever path lacks it.

## 7. Parallel is the default; serial must justify itself

One blessed flow: light thread-local scratch, per-block work, sequenced
aggregation. Heavy reusable state lives thread-local; block-locals stay
light because they travel by value. A serial loop over independent
items is a review finding — "it's the cold path" postpones the fix, and
cold paths become hot when the next feature lands on them.

## 8. Nothing is believed until measured — and nothing is measured until outputs are proven identical

Benchmark the real workload, both threadings, fresh binaries, best of
N. Identity of outputs comes first, or the benchmark measures nothing.
The bar is the wild number, not the probe. And record refuted
hypotheses next to the code — an unrecorded refutation gets
re-attempted.

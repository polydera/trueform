# Working Method

How work is orchestrated on this codebase. Follow this before any of the
per-layer pattern docs.

## Debugging

- **No fix without mechanism.** Reproduce minimal (one polygon, one loop,
  hardcoded data), instrument, name the mechanism, then fix. The fix
  replaces the workaround; it never sits on top of it.
- One hypothesis per experiment. Revert anything that does not win. Two
  guessed fixes in a row means you have no mechanism — go back and
  instrument (count it, print it, autopsy the exact failing entity).

## Performance

- Measure first: profile or count before touching anything. Know which
  stage owns the time and which counts are outsized before proposing a
  lever.
- One lever at a time, benched on the real workload, reverted if it does
  not win. Keep a record of dead ends so they are not retried.
- Structural wins first (locality, ordering, one-pass designs);
  micro-optimizations (predicates, filters) last.

## Correctness

- Correctness by construction beats correctness by audit: decide
  topology/structure before geometry can perturb it; prefer identity
  (ids, provenance) over recomputed geometry or tolerance matching.
- **Determinism is a gate**: same input → byte-identical output.
  Sequenced aggregation for parallel emission; ties broken by index,
  never by thread timing.
- Gates must be valid oracles — a gate that passes for the wrong reason
  is worse than none. Verify the binary actually rebuilt before trusting
  a green run. Test runs stay under two minutes.

## Landing

- The library stays clean until a change is proven: prototype on copies
  (experimentation/, per-language dev twins), land with the full suites
  and a regression test.
- Land small, gate, then move. Every landing updates its handoff or
  pattern doc in the same change.
- Back up any file you edit that is not yet in git.

## Delegation

- Delegate mechanical scale — file assembly from proven sources, review
  fan-outs — with exact specs that contain the acceptance criteria
  verbatim. Vague specs produce plausible garbage.
- Keep deep debugging and design decisions in your own context; they do
  not delegate well.
- Verify everything a delegate returns by re-running its gates yourself.

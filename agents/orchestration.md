# Orchestration

This document defines how the orchestrator — the one agent that holds a
whole problem, divides it, dispatches the parts and decides what lands —
solves problems in Trueform and drives the agents that build them. It sits
beside `working_method.md`: that file governs how one mechanism is
investigated, proven and landed; this one governs how a problem is divided
into mechanisms and how the work on them is distributed, reviewed and
merged. The design laws stay where they are stated — `AGENTS.md`,
`cpp_performance_philosophy.md`, `cpp_engineering_philosophy.md` — and
this file does not restate them.

## 1. Each part does one thing, and does it well

A problem is never solved whole. It is divided until each part has one
responsibility — one operation, one carrier, one output — and that part is
then made as good as it can be on its own: judged by its own carrier, its
own stated complexity and its own measurement, never by a feeling for how
much it matters to the whole. Optimal means no scan inside a loop over the
carrier, no linear search, no per-element sort; every lookup a ticket, a
dense map or one binary search into a table built once. The whole is a
composition of such parts, each reached through its accessor
(`cpp_engineering_philosophy.md` §5), and the library is bootstrapped from
its own: before a part is designed, the Trueform function that already does
it is named and read, and a part is invented only once the reason it cannot
be called is written down. What a composition replaces is deleted in the
same change (`AGENTS.md`, Implementation by subtraction).

A coupled function cannot be optimized, because no piece of it is ever
looked at alone; cannot be given to one agent, because its brief is a
pipeline; and cannot have one producer per fact, because its facts are made
in passing.

## 2. One producer, one authority

Every fact has one producer and every reader reads it there
(`cpp_performance_philosophy.md` §5). The orchestrator holds this across
agents, where it is most easily lost: two agents each building the table
they need, a mask beside a count, a fact derived in two tiers. Identities
are minted, welded and named through one authority, built before the
mechanisms that state into it. A second producer blocks landing, however
small.

## 3. Tasks

- The orchestrator decomposes; an agent executes one part. A brief that
  describes a pipeline is divided before it is sent.
- A task names one component: its contract, its carrier, the complexity it
  must have, the functions it composes, its measurement, its gate and a
  time bound. Independent tasks run in parallel in their own worktrees on
  disjoint files; the orchestrator checks the touch sets first.
- Agents are reused: the context an agent has read is the asset, a new task
  is one message to the same agent, and an agent is replaced only when it
  stalls.
- A task the orchestrator can finish in half an hour it does itself.
- Paper precedes code: the ledger (`cpp_performance_philosophy.md` §0),
  then the plan, then the build.
- An agent's proposals beyond its task are decisions for the user, never
  executed on the agent's word (`working_method.md` §7).

## 4. Instruments

- **The component ledger.** One row per header of the module under work:
  operation, carrier, parallel or serial, the complexity as written, the
  measured share, and a flag for every scan, foreign container, bare assert
  or carrier-sized serial pass, and for every scaffold — a mechanism built
  to measure that is not the landing shape — recorded the day it is
  written, with the shape it must take. The flagged rows are the debt; each
  is a task.
- **The producer audit.** One row per fact: producer, readers, any second
  producer or copy. Each duplicate is a task; at landing the audit reads
  zero.
- **The merged-tree gate.** A commit is gated on the tree it lands in,
  never on the agent's branch: cherry-picked onto the tip, every affected
  test target built there, the corpus run from that tree, then
  fast-forwarded. Statuses and volumes equal the reference, bytes are
  identical unless ruled to move, and the same at every worker count.
- **The reading.** Before a commit moves, the orchestrator reads its diff
  loop by loop. The gate proves the behaviour the corpus exercises; the
  reading proves the complexity for the input nobody ran. It is not
  delegated.
- **The stress set.** The corpus plus the synthetic scenes that stress the
  mechanism under work. The gate runs the part of it the change can reach,
  at the orchestrator's judgment; the whole set runs once at landing, never
  as an hour behind every commit.

## 5. Reports

A commit report states what changed, the complexity of every pass against
its carrier, what was deleted, what was reused and what could not be and
why, the measurement before and after, and the gate. A parallel mechanism
reports the time at one worker and at N. A report to the user ends with
the concrete state: what is green and landed, and what is beyond it.

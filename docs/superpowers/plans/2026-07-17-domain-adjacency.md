# Domain Adjacency Labels — Implementation Plan

> Working doc, untracked (house rule). Branch: `fix/domain-adjacency`.

**Goal:** answer the *adjacency* question — "which domain is on each
geometric side of this face, by its own winding" — for EVERY face,
including coincident-stack copies that carry nothing, on both paths.
Membership labels and extracted cells stay byte-identical. The client's
two tests pass as written against the new output.

**The semantic split (settled with Žiga):**
- `labels`  = MEMBERSHIP: "this face, this side, bounds this domain" —
  unchanged, still drives `split_into_domains`, keep-one emission intact
  (same-winding: min tag carries both; opposing: each member carries its
  wound-out side — the 0.9.14 provenance rule).
- `adjacency` = the canonical per-stack pair distributed to every member
  per its own winding. Same-winding duplicates: identical pairs.
  Opposing members: swapped pairs. Determined, no tie-breaks.

---

## Phase 1 — free path (`make_domain_labels`) [the client's path]

**Files:**
- `include/trueform/topology/domain_config.hpp` — add
  `domain_config::adjacency` bit.
- `include/trueform/topology/domain_labels.hpp` — add
  `tf::buffer<std::array<Index, 2>> adjacency;` (empty unless requested;
  doc: sentinel semantics for flaps, see below).
- `include/trueform/topology/domains/resolve_face_stacks.hpp` — the work:
  1. The sequential stack scan already runs `compare_faces(f, survivor)`
     for dead members; STORE the sign per member
     (`tf::buffer<std::int8_t> align_of`, +1 aligned / -1 opposing;
     survivors +1) instead of only keeping the first opposing id.
  2. In the lift, when adjacency is requested: the survivor's FULL
     lifted pair `src` (it exists in the code today, before the
     redistribution splices it) is the stack's canonical answer;
     `adjacency[f] = align_of[f] > 0 ? src : {src[1], src[0]}`.
  3. Non-stack faces: `adjacency[f] = ` their own lifted pair
     (== membership pair).
- `include/trueform/topology/make_domain_labels.hpp` — plumb the flag;
  the no-stack fast path (stacks absent) copies labels → adjacency.

**Flaps (documented limitation, phase 1):** open fragments under
`ignore_open_fragments` sit INSIDE one domain — both geometric sides are
that domain. Phase 1 leaves them sentinel in `adjacency` (containment
lives in the fuse machinery; wiring it through is a separate step, noted
as follow-up). The client's tests never touch this.

**Tests (`tests/topology/test_domain_labels.cpp`):**
- Client scenario A (two same-winding coincident planes + box,
  keep-dups, adjacency flag): BOTH planes' in-box faces report one
  inside and one outside domain; the two planes' pairs are IDENTICAL.
- Client scenario B (opposing planes): both planes report full pairs,
  SWAPPED relative to each other.
- Two-box shared-wall fixture (`make_two_box_stack`): both wall copies
  report {left box, right box}, swapped per winding.
- No-stack mesh: `adjacency == labels` element-wise.
- Flag off: `adjacency` empty; `labels`/cells byte-identical to today
  (assert against a run without the flag).

## Phase 2 — csg path exposure

The graph already computes adjacency: `part.side_label[2c + side]` per
component. Missing is only the public last mile.

**Deliverable:** `csg_domains_index_map` gains
`face_neighbor_blocks[k][j]` = the dense domain id on the FAR side of
cell k's face j (`-1` for universe/none). Computed at the same emit
sites as `face_tag_blocks` (the emitting loop/face knows its component
`c`; the far side is `side_label[2c + other]`). Dead coincident twins
never emit, so no fold lookup is needed here — attribution already
routed through `reverse_side_labels`.

**Files:** `include/trueform/csg/graph/make_csg_domains.hpp` (emit sites
+ provenance fill), `include/trueform/csg/make_csg_domains.hpp`
(index-map wrapper docs), `include/trueform/csg/csg_domains_index_map.hpp`.

**Tests (`tests/csg/test_csg_domains.cpp`):** box + plane: every wall
face's neighbor is the opposite half; hollow-pair scene: shell faces
neighbor the cavity/universe correctly.

## Phase 3 — cross-path parity + client deliverable

- Parity test: the coincident-sheets scene on both paths — free-path
  `adjacency` wall pairs vs csg `face_neighbor` across the wall must
  name the same domains (matched by volume). New test in `tests/csg/`
  or the regress harness (`experimentation/regress_domains`, section 1
  gains an adjacency check per path).
- `experimentation/repro_client_divide.cpp`: port her two tests to the
  `adjacency` array — they must pass VERBATIM in structure.
- Client reply package: corrected membership-semantics test (works on
  0.9.14 today) + note that 0.9.15 adds `domain_config::adjacency`
  answering her original assertions.

## Phase 4 — follow-ups (not this branch)

- Flap containment in `adjacency` ([D, D] for a flap inside domain D).
- Python/TS bindings: expose the flag + array (`domain_labels`
  wrapper), docs pages. Client is C++, so bindings ride the next
  bindings pass.

## Gates (house standard)

- Branch `fix/domain-adjacency`; tests land with the code.
- All 12 suites green; `regress_domains` ALL GREEN (with the new
  adjacency checks); `repro_client_divide` green.
- `-Wall -Wextra -Wpedantic` zero warnings, checked unfiltered.
- MSVC lambda checklist on every new lambda (no captured structured
  bindings, no local constexpr in lambdas) — the client builds MSVC.
- Flag-off cost: none (buffer untouched, one branch). Note in review.
- Two reviews (correctness adversarial + style/modularity), findings
  applied, then Žiga review → merge → version bump 0.9.15.

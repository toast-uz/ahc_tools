# AHC Idea Generator Skill

## Purpose
This skill is an idea dictionary for AtCoder Heuristic Contest (AHC) problem solving.
It is NOT a catalog to mechanically enumerate algorithms. Use it to detect structural signals in a problem or an existing solver, then generate concrete alternative formulations, neighborhoods, repairs, and search strategies.

## Core behavior
When asked to solve or improve an AHC solver:
1. Identify the current bottleneck: construction quality, search-space reachability, evaluator cost, feasibility, long-term dependency, congestion, ordering, allocation, uncertainty, or local optimum.
2. Select 5-10 techniques whose `triggers` match the bottleneck.
3. Include at least 2 non-obvious / reframing techniques, not only standard SA/beam/greedy.
4. For each selected technique, instantiate it for THIS problem: define state, move, invariant, evaluator, and expected benefit.
5. Prefer experiments that falsify an idea cheaply before large implementation.
6. If an existing solver is supplied, distinguish changes to representation, neighborhood, evaluator, decoder, and search controller.
7. Do not recommend a technique merely because it is famous. Explain the structural match.

## Diagnostic questions
Ask internally:
- Can time be made part of the state?
- Can the problem be solved backwards, or can an earlier decision be rewritten after seeing its consequences?
- Can part of a solution be destroyed and rebuilt exactly or semi-exactly?
- Is feasibility easier if construction and physical decoding are separated?
- Can expensive global evaluation become local/differential?
- Is the apparent decision variable the wrong representation?
- Can we reserve future resources instead of resolving collisions afterwards?
- Can a hard constraint be temporarily violated and repaired later?
- Can a sequence be cut into windows and optimized exactly inside each window?
- Can multiple weak representations/searches exchange elite solutions?
- Is there hidden symmetry, periodicity, exchangeability, or sufficient statistics?
- Can we optimize the bottleneck rather than the final score directly?

---

# Technique Dictionary

## A. State-space reframing

### A01 Time-expanded search / 時空間BFS
**Triggers:** dynamic obstacles; collisions; waiting matters; same position has different value at different times.
**Idea:** replace state `position` with `(position,time[,resources])` and search the time-expanded graph.
**Try:** add WAIT edges; cap horizon; exploit periodicity with `time mod period`.
**Failure modes:** state explosion; horizon too long; ignoring periodic compression.
**Related:** future reservation, rolling horizon, CBS, dynamic programming.

### A02 Reverse search / 逆向き探索
**Triggers:** many starts and few goals; goal constraints are tighter; predecessor generation is easy.
**Idea:** search from desired terminal states toward possible origins.
**Try:** compute reverse distances/potentials once and use them in construction/evaluation.
**Failure modes:** irreversible actions make predecessors huge or ambiguous.
**Related:** bidirectional search, potentials, goal anchoring.

### A03 Meet in the middle / 中間合流
**Triggers:** long sequence with composable halves; exponential search depth; compact boundary state.
**Idea:** enumerate/optimize prefix and suffix independently and join compatible boundary signatures.
**Failure modes:** boundary signature is too large.
**Related:** window DP, state compression.

### A04 Sufficient-state compression / 十分統計化
**Triggers:** history seems important but future depends only on aggregates; many states behave identically.
**Idea:** replace full history with counts, masks, endpoints, loads, parity, last-use times, etc.
**Prompt:** “What information from the past actually changes future legal moves or score?”
**Related:** canonicalization, transposition tables.

### A05 Symmetry quotient / 対称性潰し
**Triggers:** interchangeable agents/colors/groups; rotations/reflections; equivalent permutations.
**Idea:** canonicalize equivalent states so search budget is spent only once.
**Failure modes:** symmetry is broken by hidden future costs.

### A06 Representation inversion / 表現の反転
**Triggers:** direct variables cause awkward constraints; decoder is complicated; local moves have global effects.
**Idea:** optimize a latent representation (order, grouping, priorities, skeleton, assignments), then decode into physical actions.
**Prompt:** “What representation makes validity automatic?”
**Related:** indirect encoding, physical decoder, random keys.

### A07 Event-driven state / イベント駆動化
**Triggers:** many turns do nothing structurally; only arrivals/departures/collisions matter.
**Idea:** jump between meaningful events rather than simulating every tick.
**Related:** discrete-event simulation, lazy simulation.

### A08 Product-state search / 状態直積
**Triggers:** two small interacting constraints each easy alone.
**Idea:** search `(state_A,state_B)` explicitly instead of patching one constraint afterward.
**Failure modes:** multiplicative state explosion.

## B. Temporal reasoning and “history editing”

### B01 Retroactive modification / 過去改変
**Triggers:** current failure was caused by a small earlier choice; solution is replayable; prefix can be reused.
**Idea:** when a bad consequence appears at time `t`, identify a causal decision at `t-k`, alter it, then replay forward.
**Implementation pattern:** checkpoint -> edit historical action -> replay suffix -> accept/reject.
**Prompt:** “Instead of fixing the current state, which earlier decision would make this problem never occur?”
**Failure modes:** replay cost; cascading changes; weak causal localization.
**Related:** rollback, prefix checkpointing, suffix repair.

### B02 Delayed commitment / 遅延確定
**Triggers:** early commitment destroys options; information/value becomes clearer later.
**Idea:** leave assignments/order/route partially unresolved until necessary.
**Representations:** candidate sets, placeholders, soft ownership, partial order.
**Related:** lazy decoding, beam search.

### B03 Future reservation / 未来予約
**Triggers:** agents compete for cells/resources/time slots; collisions are expensive.
**Idea:** reserve future occupancy/capacity before executing actions.
**State:** reservation table `(resource,time)->owner/capacity`.
**Related:** time-expanded BFS, prioritized planning.

### B04 Rolling horizon / 窓先読み
**Triggers:** full-horizon optimization is impossible but short future matters.
**Idea:** optimize next `H` steps, execute only first `k`, shift horizon, repeat.
**Failure modes:** horizon boundary artifacts.
**Related:** MPC, rollout, terminal potentials.

### B05 Checkpoint & replay / チェックポイント再生
**Triggers:** simulations share long prefixes; local historical edits; expensive restart.
**Idea:** snapshot selected states and replay only from nearest checkpoint.
**Related:** retroactive modification, undo log.

### B06 Undo-log traversal / Undoログ
**Triggers:** candidate states differ locally; copying state dominates runtime.
**Idea:** record changed fields `(address,old,new)` and undo/reapply mutations rather than cloning full state.
**Related:** differential beam search, rollback DSU.

### B07 Temporal coarse-to-fine / 時間粗密化
**Triggers:** early planning needs global shape; late planning needs precise timing.
**Idea:** plan in coarse time buckets, then refine critical intervals.

## C. Construction

### C01 Greedy with regret / 後悔値貪欲
**Triggers:** assignment/scheduling; scarce options; plain greedy steals resources from constrained items.
**Idea:** prioritize item whose best and second-best choices differ most.

### C02 Randomized greedy / ランダム化貪欲
**Triggers:** greedy is strong but brittle.
**Idea:** choose probabilistically among top candidates; run many starts.
**Related:** GRASP, multi-start.

### C03 Beam search
**Triggers:** sequential construction; choices have delayed effects; branching moderate; good partial evaluation exists.
**Idea:** retain diverse top partial states instead of one greedy prefix.
**Key design:** deduplication, diversity caps, differential updates, optimistic evaluation.

### C04 Diverse beam / 多様性ビーム
**Triggers:** beam collapses into near-identical states.
**Idea:** cap children per parent/signature/region; penalize similarity; bucket by structural signature.

### C05 Scaffold-first / 骨格先行
**Triggers:** global topology matters more than local details.
**Idea:** construct backbone/tree/order/partition first, fill details later.
**Related:** decoder separation, hierarchical optimization.

### C06 Goal anchoring / ゴール固定
**Triggers:** terminal requirements are hard to satisfy after free construction.
**Idea:** fix terminal structures first and grow solution around them.

### C07 Constraint-first construction / 難制約先行
**Triggers:** a small subset of items causes most infeasibility.
**Idea:** place most constrained items first; fill flexible items afterward.

### C08 Multi-start elite construction
**Triggers:** cheap constructors produce qualitatively different basins.
**Idea:** generate many initial solutions, keep structural elites rather than only best score.

## D. Local search neighborhoods

### D01 Swap
Exchange two elements/assignments. Baseline neighborhood for permutations and grouping.

### D02 Insert / relocate
Remove one element and insert elsewhere. Often stronger than swap for ordering.

### D03 Block move
Move a contiguous segment as one unit. Useful when local subsequences already have good internal structure.

### D04 2-opt / segment reverse
Reverse an interval to remove geometric/path crossings or reorder sequence cheaply.

### D05 k-opt / edge exchange
Replace several structural edges at once when 2-opt cannot cross a barrier.

### D06 Split / merge
**Triggers:** grouping/batching/partitioning.
**Idea:** split overloaded group or merge underutilized compatible groups.

### D07 Boundary shift
Move partition boundaries rather than individual items.

### D08 Ejection chain / 押し出し連鎖
**Triggers:** moving A requires moving B, then C.
**Idea:** permit a chain of dependent relocations instead of rejecting first conflict.

### D09 Compound move / 複合近傍
**Triggers:** every useful change requires temporary worsening or multiple coordinated edits.
**Idea:** package several elementary edits as one neighborhood.

### D10 Adaptive neighborhood mix
Track acceptance/improvement per neighborhood and shift sampling budget toward productive moves without eliminating exploration.

## E. Destroy & repair

### E01 Large Neighborhood Search (LNS)
**Triggers:** local optimum; solution decomposes; a subset can be rebuilt effectively.
**Idea:** destroy a substantial region/interval/group and repair it.
**Design axes:** destroy selection, destroy size, repair solver, acceptance rule.

### E02 Window reoptimization / 区間再最適化
**Triggers:** sequence; interactions mostly local; exact solver feasible for short interval.
**Idea:** freeze outside `[l,r]`, optimize inside with DP/beam/BFS/exhaustive search.

### E03 Worst-part destroy
Destroy elements with largest marginal cost, conflicts, lateness, detours, or constraint pressure.

### E04 Related destroy
Destroy mutually related elements (nearby, same resource, same route) so repair can restructure them jointly.

### E05 Random destroy
Important as diversity baseline; avoids overfitting destroy policy to current evaluator.

### E06 Constraint repair
Temporarily allow infeasible candidate, then run a dedicated repair operator.
**Use when:** feasible-space connectivity is poor.
**Warning:** repair bias can collapse diversity.

### E07 Packet / bundle repair
**Triggers:** many atomic operations naturally travel together or share capacity.
**Idea:** repair at packet/bundle granularity instead of individual item granularity.
**Prompt:** “Which operations should be treated as one indivisible macro-action?”

### E08 Exact micro-solver
Use DP/BFS/shortest path/matching/flow/exhaustive enumeration as a repair operator on a small subproblem.

## F. Search controllers

### F01 Hill climbing
Use when neighborhood landscape is smooth and restarts are cheap.

### F02 Simulated annealing (SA)
Accept worsening moves probabilistically to cross local barriers.
**Tune:** initial acceptance rate, end temperature, schedule, neighborhood scale.

### F03 Reheating
Raise temperature after stagnation or after entering a new structural regime.

### F04 Record-to-record travel (RRT)
Accept solutions within a threshold of the best/record rather than using temperature probability.
**Useful when:** score scale makes SA temperature awkward.

### F05 Tabu search
Temporarily forbid recent moves/features to prevent cycling and force exploration.

### F06 Iterated local search / kick
Run local optimization to convergence, apply a strong perturbation, optimize again.

### F07 Multi-start
Independent searches from diverse seeds; simple and robust when basins differ strongly.

### F08 Elite pool / path relinking
Keep multiple high-quality structurally different solutions and search paths/combinations between them.

### F09 Island search
Run different parameterizations/representations/neighborhood sets in parallel conceptually; occasionally exchange elites.

### F10 Budget ladder / 予算階段
**Triggers:** need fast diagnosis and scalable search.
**Idea:** define small/medium/large fixed budgets; an idea must show signal cheaply before receiving more time.
**Useful for AI agents:** prevents spending implementation/runtime budget on weak ideas.

## G. Evaluation and bounds

### G01 Differential evaluation / 差分評価
**Triggers:** local mutation changes small part of score.
**Idea:** update only affected terms.
**Rule:** evaluator speed directly buys search iterations.

### G02 Cached local fields
Maintain distance fields, nearest-resource maps, occupancy counts, prefix/suffix aggregates, etc., incrementally.

### G03 Prefix/suffix decomposition
**Triggers:** sequence score can be composed from left/right summaries.
**Idea:** precompute prefix/suffix state so interval edits evaluate quickly.

### G04 Surrogate objective / 代理目的
**Triggers:** true score is sparse/noisy/late.
**Idea:** optimize smoother proxy correlated with final score, possibly phase-dependent.
**Warning:** periodically measure proxy-vs-true correlation.

### G05 Potential function / ポテンシャル
Add estimated future value/cost to partial-state evaluation.
**Examples:** remaining distance, flexibility, congestion risk, option count.

### G06 Optimistic bound / 楽観上界
Estimate best possible completion; prune states that cannot beat current threshold.

### G07 Pessimistic bound / 安全下界
Estimate guaranteed completion quality; useful for robust choice and pruning risky branches.

### G08 Two-stage evaluation / 二段階評価
Cheap approximate filter first, expensive exact evaluation only for survivors.

### G09 Lazy evaluation
Delay expensive score components until candidate is competitive on cheap components.

### G10 Bottleneck objective / ボトルネック代理
Instead of optimizing total score directly, identify the resource/constraint currently limiting improvement and temporarily optimize it.

## H. Speed as an algorithmic technique

### H01 Precomputation
Distances, transitions, compatibility, masks, local costs, random tables.

### H02 Bitset / bitboard
Use word-parallel operations for grid sets, reachability, compatibility, coverage.

### H03 Copy elimination
Use undo logs, persistent structures, shared prefixes, copy-on-write, indices instead of objects.

### H04 Cache-friendly flattening
Flatten grids/structures; avoid expensive division/modulo or pointer-heavy layouts in hot loops when profiling supports it.

### H05 Dirty-set update
Track exactly which cells/items/times changed and recompute only them.

### H06 Early reject
Use a cheap necessary condition or rough delta to reject hopeless moves before full evaluation.

### H07 Batch candidate generation
Generate candidates cheaply, score approximately, exact-evaluate only top subset.

### H08 Fixed-capacity structures
Avoid allocation in hot loops; reuse buffers and arrays.

### H09 Profiling-to-search conversion
Every speedup should be translated into a deliberate gain: wider beam, deeper rollout, larger LNS window, or more SA iterations.

## I. Graph, routing, allocation transformations

### I01 TSP-ification / TSP化
If the essence is visit order, explicitly convert to a routing/order problem and import 2-opt/insert/LNS ideas.

### I02 Matching-ification / マッチング化
If decisions are pairings, solve or approximate matching rather than greedy pair selection.

### I03 Flow-ification / フロー化
If resources move through capacities, model min-cost flow/max-flow locally or globally.

### I04 Assignment decomposition
Separate “who does what” from “in what order/how to execute”. Optimize layers alternately.

### I05 Tree/backbone extraction
Find a sparse structural skeleton first; optimize traversal/decoration around it.

### I06 Clustering
Group spatially/semantically related items before detailed routing.
**Warning:** clusters may create artificial boundaries; allow cross-cluster repair.

### I07 Voronoi ownership
Assign items/cells to nearest agents/seeds, then optimize boundaries.

### I08 Minimum spanning scaffold
Use MST-like structure as a cheap global connectivity prior, then modify for actual objective.

## J. Multi-agent and congestion

### J01 Prioritized planning
Plan agents sequentially; earlier routes become constraints/reservations for later agents.

### J02 Conflict-Based Search (CBS) idea
Detect conflicts, branch on which agent receives the constraint, replan locally.
**AHC use:** often approximate/truncated rather than full exact CBS.

### J03 Reservation table
Store occupied cells/edges/resources by time; route subsequent agents around reservations.

### J04 Congestion pricing
Turn shared-resource contention into a dynamic penalty; reroute agents away from overused resources.

### J05 Role assignment
Give agents differentiated roles/regions/tasks to reduce destructive competition.

### J06 Soft ownership
Prefer but do not force regions/resources per agent; allow exceptions with penalty.

### J07 Cooperative macro-action
Search joint actions for a small interacting subset while treating other agents as fixed.

## K. Uncertainty and online decisions

### K01 Rollout
For each candidate action, simulate plausible futures with a cheap policy and compare expected outcomes.

### K02 Monte Carlo sampling
Sample unknown opponent/environment parameters rather than optimizing one guessed future.

### K03 Adaptive horizon
Use long horizon when setup effects dominate; shorten when remaining time/budget shrinks.

### K04 Sequential halving
Allocate a little simulation budget to many candidates, repeatedly eliminate weaker half, spend more on survivors.

### K05 Robust objective
Optimize mean minus risk, quantile, worst-of-samples, or opponent-relative score when variance matters.

### K06 Online parameter inference
Infer opponent/environment parameters from observed actions and update simulation policy.

### K07 Exploration bonus
When information has future value, reward actions that reduce uncertainty, not only immediate score.

## L. Exact methods embedded inside heuristics

### L01 Beam-inside-SA
Use beam/DP to rebuild a local region proposed by SA/LNS.

### L02 DP window
Freeze global solution, solve short interval exactly or near-exactly.

### L03 BFS repair
Use BFS/shortest path only for the local feasibility gap instead of globally.

### L04 Matching repair
After destructive move, optimally rematch the affected subset.

### L05 Flow repair
Use min-cost flow to repair local assignment/capacity violations.

### L06 Branch-and-bound microsearch
Enumerate a small difficult subset with optimistic bounds.

## M. Meta-ideas: change the question

### M01 Causal repair / 原因を直す
**Prompt:** “The visible defect is a symptom. Which decision created it?”
Move repair point upstream rather than patching output.

### M02 Make invalid states searchable / 制約を一時的に破る
**Prompt:** “Is the feasible space disconnected under our current neighborhood?”
Allow controlled infeasibility with penalty + repair.

### M03 Optimize options, not score / 選択肢を残す
**Triggers:** early greedy gains cause late dead ends.
Reward flexibility: number/quality of future legal choices.

### M04 Change granularity / 粒度を変える
Try cell->region, item->packet, turn->phase, action->macro-action, agent->team.

### M05 Freeze the good part / 良い部分を固定
Identify stable high-quality substructure; search only unstable region.

### M06 Unfreeze the assumption / 前提を外す
List implementation assumptions not required by problem statement (fixed ordering, fixed grouping, monotonicity, one representation). Remove one deliberately.

### M07 Solve the complement / 補集合を解く
Sometimes easier to choose what NOT to use/visit/assign than what to use.

### M08 Dual viewpoint / 価格を付ける
Replace hard competition for resources with shadow prices/penalties and iteratively adjust prices.

### M09 Alternate optimization / 交互最適化
Split coupled variables A/B. Optimize A with B fixed, then B with A fixed; periodically perform joint moves.

### M10 Phase change / フェーズ分離
Use different objective, neighborhood, or representation in early/middle/late stages rather than one policy throughout.

### M11 Counterfactual replay / 反実仮想再生
Take a bad seed/state, alter one historical decision, replay, and measure causal delta. Use results to design neighborhoods.

### M12 Adversarial seed mining / 苦手seed採掘
Cluster worst seeds by failure mechanism, not just score. Create targeted operators for each cluster.

### M13 Oracle calibration / 小問題教師
Solve tiny/special instances exactly or with huge budget, compare heuristic decisions, and learn which structural choices are systematically wrong.

### M14 Search-space audit / 到達可能性監査
Before tuning SA temperature, ask whether the neighborhood can even reach qualitatively different good structures.

### M15 Decoder audit / デコーダ監査
If latent solution looks good but physical execution is bad, measure loss introduced by decoder separately from optimizer loss.

## N. Experiment design for AHC agents

### N01 Seed panel
Maintain representative seeds: average, worst, structurally distinct, regression seeds.

### N02 Ablation
Disable one component at a time; do not infer contribution from final combined score only.

### N03 Delta histogram
Compare per-seed score/move deltas, not only total average. Large regressions often reveal a new failure mode.

### N04 Fixed-budget calibration
Compare techniques at identical deterministic budgets (small/medium/large) before adaptive tuning.

### N05 Replay verification
Every saved solution/operator should replay deterministically and preserve invariants before score comparison.

### N06 Structural metrics
Track intermediate metrics (conflicts, detour, utilization, group sizes, idle time, repair count) to explain score changes.

### N07 Hypothesis ledger
For every experiment record: hypothesis, structural trigger, expected affected seeds, metric, result, interpretation, next action.

---

# Idea generation protocol

When invoking this skill on a concrete problem, produce this compact structure:

## 1. Bottleneck diagnosis
State 1-3 dominant bottlenecks and evidence.

## 2. Candidate ideas
For each of 5-10 ideas:
- **Technique**
- **Why trigger matches**
- **Concrete state/representation change**
- **Concrete move/search/repair**
- **Expected upside**
- **Cheap falsification experiment**
- **Risk / failure mode**

At least two candidates must come from section M or B unless clearly irrelevant.

## 3. Portfolio
Classify candidates:
- low-risk incremental
- medium structural
- high-risk/high-upside reframing

## 4. Experiment order
Order by `expected_information_gain / implementation_cost`, not merely expected score.

## 5. Learning extraction
After experiments, add a reusable technique or trigger if a new pattern was discovered. Prefer generalization over storing problem-specific code.

---

# Anti-patterns
- Do not say only “try SA”, “try beam”, or “tune parameters”. Specify representation and neighborhood.
- Do not tune temperature before checking search-space reachability.
- Do not optimize evaluator accuracy if evaluator speed is the actual bottleneck without measuring the tradeoff.
- Do not trust aggregate score alone; inspect per-seed regressions.
- Do not preserve implementation constraints that are absent from the statement.
- Do not use an expensive global solver when it can be embedded as a local repair operator.
- Do not assume feasibility must be maintained after every micro-move.
- Do not conflate optimizer quality with decoder quality.
- Do not keep adding neighborhoods without measuring acceptance, improvement, and unique structural reach.

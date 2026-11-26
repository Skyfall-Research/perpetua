# Changelog

Perpetua is pre-release (`0.0.1a0`): anything may change between revisions.
This file lists changes that alter results, checkpoints or the public
contract, so that a run can be compared with an earlier one.

## Unreleased

### Changes results

- **gymnax (generic), JaxMARL MPE and MEAL adapters:** the backend is reset
  from a key split off the lifetime key, as the hand-written adapters already
  did, and no longer from the lifetime key itself, which drift also derives
  its keys from. A lifetime's trajectory for a given seed changes in these
  environments.
- **PPO:** the advantage estimate skips ticks on which a slot is inactive,
  instead of using the value computed from that slot's placeholder
  observation. Runs whose slots are always active are unchanged.
- **`TorchPPO`:** the same for the PyTorch learner, which still bootstrapped
  the tick before an inactive stretch from the placeholder observation's
  value. `TorchPPO` runs on environments whose slots sit out ticks (EnvPool
  on the tick a game ends, `overlap=True`, PettingZoo agents that drop out)
  change; runs whose slots are always active are bit-identical.
  `TorchPPO.advantages(rollout)` returns the estimate.
- **DQN, IQL and SAC:** a transition whose slot is inactive on the next tick
  is no longer stored with that tick's placeholder observation as its next
  observation. It waits, collecting discounted reward, and is stored when the
  slot acts again, with the discount it accumulated. Runs whose slots are
  always active train bit-identically; the replay buffer and training state
  have new fields, so earlier DQN and SAC checkpoints do not restore.
- **`pp.evaluation.revisit`** no longer takes `key`. It was never used:
  results depend on `seeds` only. Remove the argument from calls.
- **SAC:** the two critics now start from independent parameters. Before,
  they were initialised identically and stayed identical, so clipped
  double-Q acted as a single critic. Every earlier SAC result is affected.
- **Plasticity resets** (`reset_interval`): the reset layers keep their own
  Adam state, which a reset replaces as a whole. Before, a reset zeroed their
  moments but kept the shared step count, so those layers took steps several
  times the learning rate for hundreds of updates. A reset also no longer
  reaches the CNN torso's dense layer. Runs that use `reset_interval` change,
  and their checkpoints have a new optimizer-state layout; learners without
  resets are unchanged.
- **SAC** learns only from a non-empty replay, which matters with
  `task_boundary_reset` and `update_after=0`.
- **Task switching:** task initial keys come from a tagged fold of the
  lifetime key, so a switched lifetime's trajectory for a given seed changes.
  `space_change` also fires when a task's own mask parameter drifts.
- **Drift:** `after_drift` hooks have their own random stream (they shared
  one with the regret reference policy).
- **Keys derived by the framework:** `DriftEnv` still hands the environment
  the lifetime key unchanged, but derives the drift processes' init keys and
  the drift stream from a tagged `fold_in` of it, and the `after_drift` and
  regret-reference keys from tagged folds of the step key. Before, these were
  children of a plain `jax.random.split` of those keys, so a world that split
  its own key the usual way could draw the same numbers as the hidden drift
  (BeanWorld's spawn key was the drift init key) or, with a stochastic
  reference, the reference's "random" action could equal the world's noise.
  What changes: the draws of stochastic drift processes (`RandomWalk` and the
  like) and of `after_drift` hooks, so their lifetimes differ for a given
  seed; and a stochastic regret reference (`pp.policies.random`) acts
  differently, so `regret_*` changes. What stays bit-identical: the world's
  own trajectory and the learner's metrics under deterministic drift
  (`Cyclic`, `Stationary`), and regret against a
  deterministic reference. The drift key held in the lifetime state has new
  values but the same shape, so checkpoints still restore.
- **JaxMARL MPE:** a scripted slot's random draws no longer share a key with
  the respawn reset (`num_scripted > 0`; both used `fold_in(key, 1)`).
  Lifetimes with scripted slots change; those without are bit-identical.
- **DQN, IQL and SAC with `reset_interval`:** the reset layers' initialisation
  key is no longer one the update had already used (DQN and IQL: a split of the
  key that sampled the replay batch; SAC: the next-action noise key). DQN and
  IQL now split it before sampling, so a run that resets draws different batches
  from the first update on; SAC draws the same batches and noise and only the
  fresh layers differ. Without `reset_interval` the update is bit-identical.
- **gymnax:** `BernoulliBandit-misc`, `GaussianBandit-misc` and
  `DiscountingChain-bsuite` respawn at their time limit, which is the episode
  for these tasks. Observations are float32, and the time limit is no longer
  offered as a drift field.
- **JaxMARL MPE:** observations are built from the drifted parameters
  (`view_radius` had no effect before). `simple_spread`'s scripted partner
  scores a 0.1 step, so it no longer idles most of the time.
- **EnvPool Atari:** a slot is inactive on the tick its game ends, and the
  respawn is reported with the reset frame. With `overlap=True`, only slots
  that have an observation are active at `init`.
- **BeanWorld:** items that were never collected no longer vanish for one
  regrow period at tick 2³¹. The state has a new field, so earlier BeanWorld
  checkpoints do not restore.
- **Metrics:** the window totals are re-summed every tick. The running total
  kept the float32 rounding error of one large reward for the rest of the
  lifetime, which distorted `window_mean` and `recovered_at`.
- **Metrics, `reward_variance` and `reward_std`:** the sum of squared
  deviations is accumulated with compensation, and the mean it uses comes from
  the compensated lifetime total. Before, the plain float32 sum was 4 to 18% off
  from about 4 million ticks and stopped growing near 2**24 ticks, after which
  the reported variance shrank toward zero (`healthy` stayed true). The value
  now keeps float32 precision at any tick count. `Metrics` has a new field
  (`welford_compensation`), so earlier checkpoints that contain metrics do
  not restore. `mean`, `window_mean`, `ema` and `total` are unchanged;
  `welford_mean` is now the compensated total over the count and moves in the
  last float32 bits, and `welford_m2` takes its new values. A reward that is
  constant and non-zero can report a tiny non-zero variance rather than
  exactly 0.
- **PPO and IPPO with `reward_centering`:** the TD error that moves
  `reward_rate` is computed from the same bridged quantities as the advantages.
  It used the critic's value of a placeholder observation on a tick a slot sat
  out and never saw that tick's reward, so on backends whose slots sit out
  ticks (EnvPool on the tick a game ends, `overlap=True`, PettingZoo agents that
  drop out) the rate drifted without bound. With `reward_centering` off, or every
  slot always active, results are bit-identical to before.
- **SAC with `autotune_alpha`:** the default `target_entropy` is now
  `-action_dim + sum(log((high - low) / 2))`, which is `-action_dim` for bounds
  of +-1 and the usual heuristic in normalised actions elsewhere. The policy's
  log-probability is in the environment's action units, so the old default was
  unreachable on a narrow action range (a range `high - low` narrower than about
  0.37, a box of +-0.18 or less) and the temperature grew to NaN. Autotuned runs
  on other bounds change
  (`gymnax/Pendulum-v1`, bounds +-2: from -1 to -0.31; MPE `[0, 1]`: from `-5`
  to `-8.5` for five dimensions). With `autotune_alpha=False`, an explicit
  `target_entropy`, or bounds of +-1, results are bit-identical.
- **SAC with `reward_centering`:** documented rather than changed: the rate
  settles at the average reward plus `gamma * alpha` times the policy entropy,
  so it is not the plain reward rate unless `alpha` is zero.
- **EnvPool Atari and the CNN torso:** observations are channel-last,
  `(84, 84, 4)` per slot (`(slots, 84, 84, 4)`), where they were `(4, 84, 84)`
  and the CNN torso convolved over the frame stack as an image height. The torso
  divided a frame of whole numbers in 0-255 (maximum above 1) by 255, so
  EnvPool Atari and AgarCL screen frames reached the network in [0, 1]. MinAtar
  and COOM input was passed through unchanged, so those runs were
  bit-identical. MEAL Overcooked with the CNN torso was not, which this entry
  did not say at first: three of its layers are counts, and every frame with a
  count above 1 (from the second onion in a pot until the soup is delivered)
  was divided by 255 as a whole. An AgarCL grid frame without a -1 cell was
  integer-valued and was scaled too. The next entry replaces this per-frame
  rule.
  Earlier EnvPool Atari results and AgarCL runs with the CNN torso are not
  comparable. The EnvPool path was checked only
  with synthetic frames of the shape EnvPool returns; EnvPool does not run on
  macOS.
- **CNN torso (PPO, IPPO, DQN, IQL): one pixel scale per learner.** The torso
  divides every observation by one number that is fixed when the learner is
  built, where it decided frame by frame from the frame's values. Build the
  learner with `observation_space=env.observation_space` and the scale is 255
  for a `Box` with bounds of 0 and 255 in every entry (EnvPool Atari, a
  Gymnasium `uint8` image environment, AgarCL's screen) and 1 for any other
  space; `PPOConfig(pixel_scale=...)` and `DQNConfig(pixel_scale=...)` set it by
  hand. It is `agent.pixel_scale`, and `runner.metadata["learner"]["pixel_scale"]`.
  What changes: **MEAL Overcooked with the CNN torso**, on every frame that
  holds a count above 1, which is no longer divided by 255 (the scale is 1 on
  every tick); an EnvPool Atari frame whose pixels are all 0 or 1, which is now
  divided like the others; an AgarCL grid frame without a -1 cell, which is no
  longer divided. What stays bit-identical: MinAtar, gymnax Pong and COOM runs,
  MEAL frames with no count above 1, and raw 8-bit frames with a pixel above 1
  (compared with the old torso on synthetic frames, macOS CPU; EnvPool, COOM and
  AgarCL do not run there). Of the recorded baselines only
  `ppo/cnn-meal-pot-fill` moved (its `after` frame); the short MEAL training
  baselines never fill a pot and are unchanged. A CNN learner built without
  `observation_space` or `pixel_scale` has scale 1, and a runner refuses it on
  an environment whose space gives 255, so an EnvPool Atari script needs the
  argument. `PPOConfig` and `DQNConfig` have a new field, which the recorded
  learner configuration includes, so earlier PPO, IPPO, DQN and IQL checkpoints
  are refused as a configuration mismatch.
- **k-percent tuning:** `tuning_ticks` no longer overshoots by one tick
  through float rounding.
- **`compiled=True` and `initial_active`:** CPU lifetimes honour a backend's
  `initial_active`. With `EnvPoolCartPole(overlap=True, compiled=True)` the half
  of the slots that has no observation yet is inactive at `init`, as on the
  eager path, instead of every slot being active on the first tick, so the
  first tick and the metrics that follow from it change for those runs.
- **Tuning selection `"iqm"`** now ranks candidates by the mean of the middle
  half of their sorted scores, with fractional weight at the cuts. It used
  `reporting.interquartile_mean`, which trims `n // 4` scores per end and so
  trimmed nothing with fewer than four tuning seeds: the preset default
  (`DEFAULT_TUNING`, 3 seeds, `"iqm"`) selected exactly as `"mean"` does. With
  three seeds the selected candidate can differ now.
- **Checkpoints** have format version 2. The saved structure record no longer
  holds `str(treedef)`, which printed a memory address for a pytree node with
  a static field and made that checkpoint unrestorable in a new process; it
  lists the key path of every array leaf and the type name of every
  container. Checkpoints written before this change are refused with a message
  that says so.
- **Task switching, respawns:** a switch reports `info["respawn"]` from each
  task's own `observe`, where it read a `respawn` attribute of the task state.
  `PartnerGrid` and `OpponentGrid` (which keep it one level down) never
  reported a respawn inside a switch before; a custom environment whose state
  has no `respawn` field now does too. An episode that ends on the tick of a
  switch is reported on that tick (native switches dropped it; host switches
  reported it late, on re-entry, next to another task's transition).
- **Task switching, checkpoints:** a switch records each task's own
  configuration (and order) instead of the bare ids, so earlier checkpoints of
  switched lifetimes are refused, and a restore into a switch with swapped or
  reconfigured tasks is refused instead of continuing differently. A switch is
  marked non-checkpointable when any task is (a `compiled=True` simulator),
  so `checkpoint.save` refuses it and `check_env` skips the checkpoint check.
  `pp.presets.continual_world` also records `seed`, `max_episode_steps`,
  `compiled` and `randomize_goals` in its preset block, and each Meta-World task
  records its `seed`.
- **MORPHEUS:** the Oracle's `regime_index` and `task_id` name the regime
  declared on the tick of a switch, as `task_id` does elsewhere. They named the
  previous regime for one tick before. Protocol labels built from them move by
  one tick at each switch. `chaos_active` is false during the baseline revisit
  (it was true). The live client reads every page of orders, invoices and
  delivered shipments, where it read the first page only, so units, throughput
  and ledger rewards of a long-lived world change. Each tick now makes one
  request per page of those records. Tested against the mock only.
- **`examples/compare_benchmarks.py`:** the host evaluation score is the mean
  reward over all slots, as in training, and no longer slot 0 alone. Evaluation
  scores of tasks built with more than one slot (the Atari tasks, `num_envs=2`)
  change; single-slot tasks are unchanged.
- **`examples/measure_switching.py`:** the `switching` timing waits for the
  compiled runner's result before reading the clock, so its
  `ticks_per_second` for the MinAtar preset changes (it could stop the clock
  before the device had finished).
- **`scripts/reproduce_continual_world.py metrics`** leaves out a continual
  run whose `done` record is missing or reports `complete: false`, and names
  it on stderr as aborted. Before, a run that turned unhealthy in its last
  chunk, after writing its final evaluations, was averaged in. Metrics computed
  from such a log change.
- **`scripts/probe_gymnax.py`:** the `float` column counts the fields the
  adapter can treat as float parameters; it counted integer fields and the
  time limit as well.
- **SAC probes:** `SAC.probe` feeds the critic the policy's mean action
  squashed and scaled to the action bounds, as training does; it fed the raw
  pre-tanh mean. `dormant_critic` and `rank_critic` change for every SAC run,
  including the histories a `ProbedRunner` records. Training and the actor
  probes are unchanged.
- **PPO `reward_centering` with slots that sit out ticks:** the TD error
  bridged across an inactive stretch is divided by the discounted number of
  ticks it spans before it moves `reward_rate`. Before, the effective step was
  `centering_step_size` times that count, and the estimate diverged once the
  product exceeded 2 (step 0.1, `gamma` 0.99, a slot acting every 32nd tick).
  What changes: PPO and IPPO runs with `reward_centering=True` in which a slot
  is inactive on some tick (`reward_rate`, and through it the advantages).
  What stays bit-identical: runs without `reward_centering`, and centered runs
  whose slots are always active, under compute-aware time included. DQN, IQL
  and SAC centering is unchanged.
- **`OpponentGrid-v0` with two or more opponents:** opponents spawn on
  distinct cells (they were drawn with replacement), and when only opponents
  propose the same free cell the lowest slot among them moves instead of all
  of them staying. Before, two opponents at aggression 1 that shared a cell,
  or kept proposing the same cell, never moved again. What changes: every
  lifetime with `num_opponents >= 2` (initial positions and movement), and the
  `movement` string of its recorded configuration. What stays bit-identical:
  lifetimes with `num_opponents=1`, the default, with any number of learners.
- **`pp.evaluation.revisit` on worlds with scripted or inactive slots:** the
  score averages the per-tick mean reward over the learner slots (those whose
  `active_mask` was true at the start or after any tick of the evaluated
  lifetime), not over every slot. On the zero-sum `OpponentGrid-v0` the
  all-slot average was 0 for any policy, and `forgetting` built on it was
  meaningless. What changes: scores on `PartnerGrid-v0` and
  `OpponentGrid-v0`, whose scripted slots never act, and on any other
  lifetime in which some slot never acts. What stays bit-identical: single-agent worlds and worlds in
  which every slot acts at least once.
- **`pp.probes.ProbedRunner` on worlds with inactive slots:** the probe batch
  leaves out rows whose slot was inactive (placeholder observations), by
  giving them weight 0 in `dormant_unit_ratio` and `effective_rank`. On
  `PartnerGrid-v0` with one learner and three partners, 24 of 32 rows were
  zeros. What changes: probe readings whenever a slot in the probe batch was
  inactive. What stays the same: readings of runs whose slots are all active
  (equal on CPU for PPO, DQN and SAC), and training, which probing never
  touches. `ProbeTrace` has a new `active` field, so earlier `Probed`
  checkpoints do not restore.
- **`AECLifetime`:** an agent that terminates while an earlier agent in the
  cycle stays alive is inactive from that tick and is sent `None` when the
  selector reaches it. Before, it was reported active for one more tick and
  the next tick raised "when an agent is dead, the only valid action is
  None". An agent that dies before its turn in a cycle is now skipped (this
  raised before), and a live agent that reports `active_mask` false is sent
  action 0 instead of the policy's action, as the parallel lift does. What
  changes: lifetimes in which those cases occur, which either raised or acted
  for a slot that was not acting. What stays identical: lifetimes in which
  agents die only at the end of a cycle with the selector on a dead agent,
  the MPE2 lift included.
- **`PettingZooLifetime` over a lifetime-preserving view**
  (`PettingZooAdapter(env, horizon=H)` lifted again): the end of a synthetic
  horizon no longer sets `respawn` for every slot. The flags are the wrapped
  world's own, so with `horizon=2` they are no longer true on every second
  tick. What changes: `info["respawn"]` of such lifts, and anything that reads
  it (SAC `mask_respawns`). What stays identical: observations, rewards and
  activity of those lifts, and every lift of an ordinary PettingZoo
  environment, whose reset restarts the legacy world.
- **SAC log-probability:** the Gaussian term is computed from the sampled
  noise, not from `(pre_tanh - mean) / std`, which cancels in float32 once the
  standard deviation is near the spacing of floats around the mean (log-std
  below about -14; the clamp is -20). In a four-sample check at log-std -17
  the old form was off by up to 0.39 per sample, and its gradient with
  respect to log-std summed to -3.50 where the exact value is -4. What
  changes: every SAC run, in the last float bits
  at ordinary standard deviations and materially for a nearly deterministic
  policy. Nothing in SAC stays bit-identical; PPO, DQN and IQL are untouched.
  This supersedes the other entries in this section that call a SAC run
  bit-identical: each of those holds for its own change only.
- **SAC `reset_last_layers`:** the actor's mean and log-std heads count as
  one layer, so the count reaches the same depth in the actor and the
  critics. Before, 1 reset the log-std head only and the default 2 reset the
  actor's two heads but each critic's last hidden layer and head. What
  changes: SAC runs with `reset_interval` set (which actor layers a reset
  replaces, and the layout of the actor's optimizer state, so earlier
  checkpoints of such runs do not restore); the largest accepted count is
  now `hidden_layers + 1` for the actor too. What stays
  bit-identical relative to the entry above: SAC without `reset_interval`.
  PPO, IPPO, DQN and IQL selection is unchanged.
- **SAC `mask_respawns` with slots that go inactive:** a transition that
  waits while its slot is inactive is masked when the slot was flagged
  `respawn` on any tick since it acted, not only on the tick that completes
  it. What changes: SAC runs with `mask_respawns=True` on environments that
  flag a respawn while the slot is inactive (custom or wrapped environments;
  the bundled adapters flag it on the tick the slot becomes active again).
  What stays the same: runs without `mask_respawns`, and runs whose slots are
  always active. The SAC training state has a new field
  (`pending.respawned`), so earlier SAC checkpoints do not restore.
- **`mjx/CartPole-v0` and `mjx/Pendulum-v0` under `mass_scale` drift:** the
  constants MuJoCo derives from the masses are scaled with them (subtree
  masses and `dof_M0` by the scale; `dof_invweight0`, `body_invweight0` and
  `actuator_acc0` by its inverse), so the stepped model equals one recomputed
  with `mj_setConst`. Before, only `body_mass` and `body_inertia` changed, and
  the joint-limit constraint used the unscaled inverse weights: at
  `mass_scale` 4 the cart at its slider limit was up to 0.02 off a recomputed
  model over 300 decisions under a constant push, and is within 1e-5 now.
  What changes: lifetimes whose `mass_scale` differs from 1, visibly once a
  joint limit is active (CartPole's slider; the pendulum has no limit). What
  stays bit-identical: lifetimes without `mass_scale` drift, gravity drift
  included (checked on macOS CPU).
- **Task switching under a gradual schedule:** `pp.tasks.switch` declares a
  switch on every tick on which the active task changes, whatever process
  drives `task`. Before, only the process's own phase change counted, so a
  `RandomWalk` (alone or inside a `Sum`) changed the task, restarted it and
  raised `space_change` without ever setting the Oracle's `boundary` or
  `task_changed`. Lifetimes whose `task` is driven by such a process now
  report those flags, run recovery tracking (`has_switch`, `since_switch`,
  `recovered_at`) and trigger SAC's `task_boundary_reset`; their trajectories
  under a fixed policy are unchanged. Table schedules (`Cyclic`, `Schedule`,
  `Sequence` of tables) are bit-identical, including a table with repeated
  values, whose phase change still counts as a switch.

### Added

- `pp.testing.reward_action_sensitivity(env, seed=, ticks=, actions=)`: the
  fraction of ticks, along a random lifetime, on which the reward differs
  between valid actions stepped from the same state. Where it is 0.0,
  same-state regret (`reference=`) is zero for every pair of policies.
  `scripts/survey_regret.py` runs it over every registered JAX environment.
- Recorded baselines for seeded runs. `scripts/record_baselines.py` runs 27
  short seeded lifetimes (a random policy on six environments, and PPO, IPPO,
  DQN, IQL and SAC with their options for a few updates) and writes a digest of
  each to the fixture of the platform it runs on
  (`tests/fixtures/baselines-darwin-arm64.json`,
  `tests/fixtures/baselines-linux-x86_64.json`); `tests/test_baselines.py`
  recomputes them and fails when one differs from the fixture of its own
  platform, so a change to seeded results that is missing from "Changes
  results" is noticed. The comparison is exact on macOS arm64; on Linux x86_64
  booleans and integers are exact and floats may differ by a relative 5e-5
  (`LINUX_RTOL`) plus an absolute 5e-7, five times the measured spread
  between two runner CPUs, in place of the earlier 5e-2 relative-and-absolute tolerance
  that hid a 5% change and any change to a number below 0.05. A platform
  without a fixture skips the comparison; CI on Linux x86_64 fails instead.
  Every digest also holds checksums of the final observation, and the
  random-policy digests checksums of every action and observation, so a run
  whose reward is always zero (MEAL Overcooked) no longer has a digest that
  cannot move. The Linux fixture is recorded by the `Record baselines`
  workflow (`.github/workflows/baselines.yml`), and
  `scripts/record_baselines.py --compare` prints the largest relative
  difference between a recording and a fixture.
- `pp.probes.dormant_unit_ratio` and `pp.probes.effective_rank` take optional
  row `weights=`; a row of weight 0 is left out at a fixed batch shape. The
  `probe` method of PPO, DQN, IQL and SAC takes the same `weights=` and passes
  them on.
- `reporting.stratified_bootstrap_ci` no longer returns a lower bound above the
  upper bound. When every resample is equal, interpolating the two percentiles
  rounded them one ulp apart on a GPU; the upper bound is now at least the lower.
- Perpetua is released under the MIT licence (`LICENSE`). `THIRD_PARTY_NOTICES.md`
  records the licence of each upstream project, the README acknowledges them, and
  the proposal slides are in `docs/slides/`.

- `scripts/generate_env_docs.py --check` lists the pages that would change and
  exits 1 if any would, without writing.
- `Oracle`, `OneHotObservation` and `pp.compose` on a wrapped environment say to
  compose before wrapping, naming the wrapper. `Metadata` has an optional
  `forkable` key (False for the single-use state of a compiled bridge).
- `pp.environments()` lists every id `pp.make` accepts (the built-in ids, then
  third-party ones). `pp.registered()` still lists third-party ids only.
- `pp.environments(builtin=True)` lists the built-in ids only.
- `pp.spec(id)` returns an `EnvSpec` for a built-in id without constructing
  it: `namespace`, `extra`, `optional_extra`, `host`, `platform` and `setup`.
  `pp.make`, the environment pages' install rows, the soak script and the
  tests read these facts from it; they were written separately in each.
- `perpetua.agents` exports `PPOHyperparameters`, `DQNHyperparameters` and
  `SACHyperparameters`.
- `HttpMorpheusClient` warns when it is built with a credential header and a
  plain `http` address that is not the local machine.

- The training examples (`train_ppo.py`, `train_dqn.py`, `train_sac.py`) print
  progress lines to stderr, take `--quiet`, and describe every option in
  `--help`. `examples/README.md` says what each example is for.
- `EnvPoolAtari`'s `task` defaults to `"Breakout-v5"`, as
  `pp.make("envpool/Atari-v0")` already did.
- `PartnerGrid` has a `size` attribute, as the other grids do, so
  `pp.make("PartnerGrid-v0").size` works.
- A parameter environment may define `check_drift(drifts)`. `DriftEnv` calls
  it whenever drift is bound (construction, `with_drift`, `pp.compose`), so
  the environment can refuse a process it cannot run.
- `python -m perpetua.adapters.coom_setup` and
  `python -m perpetua.adapters.agarcl_setup` build the COOM and AgarCL
  runtimes from an installed package. The "runtime not found" errors name
  these commands; `scripts/setup_coom.py` and `scripts/setup_agarcl.py` remain
  as wrappers around them.
- The package metadata carries trove classifiers (Python 3.12 and 3.13), and
  `.github/dependabot.yml` asks for one grouped update a month for the
  workflow actions and for the locked Python dependencies.
- `pp.reporting.provenance()` returns the code and software behind a result as
  a JSON-ready dict: Perpetua version, source revision (the commit of a git
  checkout with a `dirty` flag, or the commit of an install from a git URL,
  else None), JAX version, backend, Python and platform. It includes no path,
  host or user name. Every example and script that prints a JSON report
  carries it under a new `provenance` key (`examples/train_*.py`,
  `continual_grid.py`, `control_ppo.py`, `compute_ppo.py`,
  `compare_benchmarks.py` and the `measure_*` examples; `scripts/soak.py`,
  `learning_guard.py`, `check_runtime.py`, `claim_figures.py`,
  `coom_lifetime.py`, the `measure_*` scripts, and the run header and metrics
  summary of `reproduce_continual_world.py`); their existing keys are
  unchanged. `scripts/measure_baseline.py` reports `null` when it runs where
  Perpetua is not installed. `scripts/record_baselines.py` (the regression
  fixture) and `scripts/plot_claim_figures.py` do not carry it.
- A wheel or sdist built from a git checkout records the commit it was built
  from and whether tracked files had uncommitted changes, in a generated
  `perpetua/_revision.py` (written by the build hook `hatch_build.py`, never
  into the source tree). `pp.reporting.provenance()` reads git first (a
  checkout or editable install reports its live state), then the commit an
  installer recorded for a git URL, then this record. Before, the documented
  `git clone` and `pip install ".[training]"` gave `revision: null`. A build
  outside a checkout still works and records nothing.
- Checkpoints record the same provenance next to the JAX, Orbax and backend
  entries, and `pp.checkpoint.provenance(path)` reads the record back. The
  format version is unchanged: earlier version-2 checkpoints restore as
  before. `restore` compares the recorded Perpetua version, revision, JAX,
  Orbax and backend with the running ones and emits one `UserWarning` listing
  the differences; it does not refuse. Runner `metadata` does not contain the
  provenance, so a checkpoint written at another commit is not refused for it.
- `pp.runtime.ChunkRunner` writes the runner contract down as a protocol:
  `runner(carry)` returns the next carry, `metadata` is the configuration, and
  the read-only `donates` says whether the call consumes its input carry. All
  twelve runner classes have `donates`; `ProbedRunner` forwards it from the
  runner it wraps, and `probes.LearnerRunner` extends the protocol. The two
  compute-aware runners have `metadata` and `donates` but take a timing trace
  as a second argument.
- Parameter environments may define `declared_switches(previous, current)`,
  which flags drifted fields as switched on a tick in addition to their
  processes' own phase changes, and `check_schedule(field, data)`, which
  `assign_schedule` calls. `TaskSwitch` defines both.

### Changed

- `pp.checkpoint.restore`'s software warning also lists a revision that only
  one side knows, and uncommitted changes on either side (`dirty` true): a
  checkpoint written from a dirty tree at the same commit restored silently.
  It is still one `UserWarning` and never a refusal, and its first words are
  now "This checkpoint is not known to come from the running software".
  Restoring in a checkout with uncommitted changes therefore always warns.
  Checkpoints from before revisions were recorded are compared as before.
- **Recovery tracker:** a switch reported on the first `metrics.update`,
  before the window holds any reward, is no longer tracked. It took a baseline
  of 0.0 from the empty window, so a non-negative reward stream read as
  recovered after `window_size` ticks and a negative one never. Only a
  process that switches on tick 0 does this (a `Cyclic` drift of period 1
  among the built-in ones), and there `has_switch` now turns true one tick
  later; no recorded baseline changes. `since_switch` stops at `2**31 - 1`
  instead of wrapping negative; it keeps its int32 dtype, so checkpoints are
  unaffected.
- `pp.reporting.stratified_bootstrap_ci` (and `report`) always return
  `low <= point <= high`: a bound the percentile interval leaves on the wrong
  side of the point is moved to the point. For constant scores the upper bound
  could sit one float32 step below the point (0.10000000149 against
  0.10000000894); a skewed statistic on few runs can move a bound further.
  Intervals that already contained their point are unchanged.

- `HostDQNRunner` and `HostSACRunner` write each tick's transition into the
  replay buffer in place. They ran the insert uncompiled, which copied every
  replay array on every tick, so a tick cost time proportional to
  `buffer_capacity` (about 50 ms for one slot of 4,096 `(84, 84, 4)` frames
  on a Mac CPU). A call now copies the replay once, which keeps the carry
  passed in readable (`donates` stays `False`), and the remaining ticks reuse
  that copy. Results are bit-identical.
- Every public module declares `__all__`: the names the documentation,
  examples and tests use, and the types in their signatures. Helpers that only
  other Perpetua modules use are still importable by name but are no longer
  part of `from module import *` or of the API reference; among them are
  `adapters.bridge.CallbackWorld`, `adapters.mpe2_snapshot.MPE2Codec`,
  `adapters.snapshots.CartPoleCodec` and `params.param_paths`. The network
  modules (`ActorCritic`, `QNetwork`, `GaussianPolicy`, `ContinuousQNetwork`)
  stay public. `perpetua.envs` is a regular package.
- `HttpMorpheusClient.seen` is bounded: it holds the tickets open at the last
  read and the last 10,000 that left the open list, as an insertion-ordered
  dict (it was a set that grew for the whole lifetime). A ticket listed as
  open again after 10,000 others have left counts as a new failure again.
  The live client is still tested against a mock deployment only.
- `pp.make("metaworld/...")` and `pp.make("envpool/...")` name the extra to
  install when a third-party module is missing, as the other optional
  families already did.
- `pp.checkpoint.replay` with a donating runner inside `ProbedRunner` returned
  deleted arrays for every chunk but the last, because it read a private
  attribute the wrapper did not have. It now reads the public `donates`, so
  every returned carry is readable for bare and wrapped runners. No run's
  numbers change.
- A switch has one name in the metrics too. The carry fields are
  `switch_baseline`, `has_switch` and `since_switch`; `metrics.update` takes
  `switch=`; the helper is `runtime.drift_switch`. The old names
  (`change_baseline`, `has_change`, `since_change`, `change=`,
  `runtime.drift_change`) still work and warn. `recovery_ticks` names its
  second argument `switches`; it was `changes`, with no alias, so pass it by
  position. Checkpoints are unaffected: the field order did not change.

- `pp.drift.Schedule` keeps its table in a `ScheduleTable` and
  `pp.drift.Sequence` its segment bookkeeping in a `SegmentCounter`, where both
  reused `DriftState` with other meanings for its fields. Values are unchanged.
  A `Sequence` state has one leaf fewer, so earlier checkpoints of lifetimes
  that use `Sequence` (including `pp.presets.morpheus_regimes`) do not restore.
- **One name for a switch.** The Oracle field for "a declared switch happened
  on this tick" is `boundary`, for drift and task sequences alike.
  `drift_changed` holds the same value and is deprecated: exposing it through
  `pp.Oracle` raises a `DeprecationWarning`. The per-parameter
  `{field}_changed` fields are unchanged.
- `jax_enable_x64` no longer breaks the hand-written gymnax CartPole and
  Pendulum wrappers (observations stayed float64 against a float32 space),
  `mjx/*` lifetimes and `SACRunner` (a float64 action in a float32 carry).
  `gymnax/FourRooms-misc` takes Perpetua's int32 actions under the flag.
- `gymnax/Pendulum-v1` is made without the two Gymnasium precision warnings.

- DQN, IQL, PPO, IPPO and SAC convert hyperparameter overrides to float32
  arrays in `init`, so a Python float or NumPy override compiles the runner
  once and restores into a template built the same way. Before, the state held
  the Python or float64 value, the runner compiled twice and `checkpoint.restore`
  failed with a structure mismatch.
- The native worlds (`ContinualGrid`, `DoorGrid`, `ForagerGrid`, `OpponentGrid`,
  `BeanWorld`) and SAC run under `jax_numpy_rank_promotion="raise"`. Results
  are bit-identical.
- SAC builds its default configuration under `jax_debug_nans`; an unset
  `target_entropy` is no longer stored as NaN.
- Documented limits: a PPO slot that is inactive when a rollout ends bootstraps
  its last action from the critic's value of the placeholder observation; and
  the in-carry recovery tracker (`recovered_at`) pools every agent slot, so it
  is not meaningful on zero-sum `OpponentGrid` or with scripted slots.
- `examples/measure_training_throughput.py` reports the lifetime's mean reward
  per tick as `lifetime_mean_reward_per_tick_last_seed`; the key was
  `mean_episode_return_last_seed`, although a lifetime has no episodes.
- The source distribution (sdist) no longer includes `.github/`,
  `.test_durations` or `docs/slides/`; the wheel is unchanged.
- The documentation tools are a locked dependency group: build the site with
  `uv run --group docs sphinx-build -W docs _build/html`. The build no longer
  suppresses autodoc warnings, so a directive that fails to import fails it.
- `scripts/soak.py` no longer stops with a `TypeError` in its host and worker
  modes when a worker's memory could not be read because it had exited.
- **Compiled bridge (`compiled=True`):** consecutive simulator calls are
  ordered even when the actions do not depend on the previous observation.
  The host callback now returns the next generation, which the next call
  consumes; before, the generation was computed outside the callback, so
  open-loop steps unrolled in one program (a Python loop under `jit`, or
  `lax.scan(unroll=k)`) could reach the simulator out of order and be refused
  as "stale or forked". Runs that completed before are bit-identical.
- A worker request (COOM, AgarCL) honours its timeout while sending as well:
  a worker that stopped reading part-way through a large request blocked the
  caller indefinitely. The COOM and AgarCL workers no longer close an
  environment twice after a failed re-init.
- EnvPool: a lifetime seed in the top `num_envs - 1` values of the 31-bit
  range is folded into `[0, 2**31 - num_envs]`, because EnvPool adds the slot
  index to it and the native snapshot pool refuses a larger one. Every seed
  that was accepted before is passed on unchanged. Integer actions reach the
  pool as a C-contiguous `int32` array; a strided view was refused by the
  snapshot pool.
- `MorpheusLifetime`: a refused second `init` no longer changes the regime the
  adapter believes the world is in, and an invalid action is refused before
  the regime change is sent, not after.
- Presets validate `period` before building simulators and close the ones
  they built when the switch is refused (`on_switch="restart"` on host lifts,
  for example), which leaked them before. A NumPy integer `period` is stored
  as a Python int, so the preset metadata stays JSON-serialisable and
  `checkpoint.save` accepts it.
- `TorchRunner` passes the agent's `act` the whole public info: `respawn`,
  `space_change` and the tick `t` (an int64 scalar) next to the two masks, and
  under an `Oracle` the exposed fields as `info["oracle"]`. Before, a PyTorch
  agent received only the masks although the exposure was recorded.
  `TorchPPO` reads only the action mask, so its runs are unchanged.
- `OneHotObservation` returns float32 observations, as its space declares,
  whatever the wrapped observation's dtype. A field value outside
  `[0, size)` encodes as all zeros also when it is a float in `(-1, 0)`, NaN
  or infinite; those could encode index 0. Integer fields in
  range and float32 observations are unchanged.
- Compute-aware stepping (`compute.advance`, `ComputeRunner`) no longer marks
  the lifetime unhealthy when an inactive slot has no valid action; only an
  active slot needs one, as the environment contract says.
- `TimingTrace.from_ticks` and `compute.record_trace` convert NumPy integer
  scalars before multiplying or shifting, so a 32-bit `quantum_ns` or cost no
  longer wraps or raises a spurious `OverflowError`.
- `reporting.stratified_bootstrap_ci` returns NaN bounds whenever the point
  estimate is non-finite; with few bootstrap draws the bounds could be finite.
- `probes.dormant_unit_ratio` and `probes.effective_rank` return NaN when any
  activation is non-finite (they reported 0 and 1). `effective_rank` no longer
  floors the sum of singular values at `1e-12`, so activations of very small
  scale keep their rank.
- `evaluation.recovery_ticks` accepts switch indices of any integer dtype:
  unsigned indices no longer wrap in the reference window, and narrow ones no
  longer raise a false range error on a long trace.
- `pp.environments()` lists a built-in id once after it is overridden with
  `register(..., overwrite=True)`.
- `measurement.code_lines` takes comments from the tokenizer: a line starting
  with `#` inside a docstring counts as docstring, and inside any other string
  as code. Both counted as comments.
- `requires-python` is `>=3.12,<3.14` (it was `>=3.12`). The locked MuJoCo
  3.3.0 has no wheels for Python 3.14, and TensorStore does not build on 3.15.
  CI now also installs the locked environment on Python 3.13 and runs the
  contract, conformance, checkpoint and PPO tests there; `.python-version`
  makes `uv sync` choose 3.12, the version the full suite runs on.
- MEAL's dependencies (`meal-bench`, `distrax`, `tfp-nightly`) moved from the
  `dev` dependency group to a `meal` group. Both are installed by default, so
  `uv sync` is unchanged; `uv sync --no-group meal` leaves MEAL out, and its
  tests are then skipped. The core alone is now
  `uv sync --no-default-groups` (`--no-dev` still installs the `meal` group).
- CI runs on `ubuntu-24.04` with every action pinned to a commit and uv pinned
  to 0.11.23. The torch jobs install the lockfile's torch version (as a CPU
  wheel) with its locked dependencies, and the wheel job fails if a snapshot
  test is skipped.
- Tests that need PyTorch carry a `torch` pytest marker (`tests/conftest.py`
  skips them without torch). The torch jobs run `pytest -m torch` and fail if
  a selected test skips; three torch-gated tests that no job ran are now
  among them. The `torch` job of `learning.yml` is renamed `torch-guard`, so
  the required check `torch` names one job.
- Without the `meal` group (`uv sync --no-group meal`) the eight tests that
  build MEAL Overcooked skip with that reason; they failed before.
- `python -m perpetua.adapters.coom_setup` and
  `python -m perpetua.adapters.agarcl_setup` recreate the runtime's
  environment when its interpreter is missing, dangling or does not start,
  instead of failing with "a directory already exists". The CI caches of both
  runtimes are keyed by the workflow file as well, so a cache made under
  another uv setup is not restored.
- `uv.lock`: Pillow 12.3.0 to 13.0.0 and opentelemetry-api, -sdk and
  -semantic-conventions 1.45.1 to 1.45.2 (0.66b2), for the security fixes in
  those releases. No other locked version changes.
- Dependabot no longer proposes versions of the pins that move together or
  are held on purpose (`mujoco`, `mujoco-mjx`, `gymnax`, `metaworld`,
  `meal-bench`, `distrax`, and major or minor releases of `jax` and `jaxlib`).
- Package metadata has `authors`, `keywords` and a `Documentation` URL (the
  repository's `docs` folder).

### Stricter checks

- `pp.checkpoint.save`, `restore` and `replay` refuse a `TorchRunner` and its
  `TorchTrainingState` with a `TypeError` that names `TorchRunner.save` and
  `TorchRunner.restore`. The PyTorch agent's networks, optimizer and sampling
  state are not in that carry, so the generic functions wrote a checkpoint
  without them and `replay` continued from whatever the agent held, with no
  error. `TorchRunner.save` and `restore` are unchanged and exact.
- `ComputeRunner` and `ComputePPORunner` refuse a timing trace whose tick
  counts are not `tick_count(cost, quantum_ns)` for the runner's own
  `quantum_ns`, which is what a trace built with another quantum looks like.
  It was replayed, and the charged `compute_ns` disagreed with the ticks run.
  A zero tick count still marks the carry unhealthy instead of raising.
- `pp.checkpoint.restore` compares the shapes and dtypes of the arrays
  actually stored with the template, from Orbax's record of them and before
  loading. A state directory placed under another checkpoint's header with
  the same shapes and another dtype was cast into the template silently.

- **PPO and IPPO** refuse a rollout that leaves fewer than two samples per
  minibatch (`rollout_steps * agents == minibatches` for PPO,
  `rollout_steps == minibatches` for IPPO, for example IPPO with
  `rollout_steps=4` and the default four minibatches). The advantage is
  normalised per minibatch, so a single sample normalised to exactly zero: the
  policy loss was 0.0, the actor moved only through the entropy bonus, and the
  run stayed healthy. The runners refuse it when they are built and
  `PPO.update` when it is called, as `TorchPPO` already did.
- **Float settings of `PPOConfig`, `DQNConfig` and `SACConfig`** accept Python,
  NumPy and JAX real scalars and are stored as Python floats, as the integer
  settings already are. A `np.float32` learning rate or a `jnp` gamma trained
  but made the first `pp.checkpoint.save` fail, because the recorded
  configuration was not JSON. Booleans (`learning_rate=True`), strings and
  arrays with dimensions are refused with the type and value received. The
  traced `*Hyperparameters` are unchanged.
- `SACConfig` refuses `rollout_steps` above `buffer_capacity`, with the error
  `DQNConfig` gives.
- **Pixel scale:** `PPORunner`, `HostPPORunner`, `DQNRunner`, `HostDQNRunner`
  and `ComputePPORunner` check the CNN learner's derived pixel scale against
  the environment's space when they are built, not only in `init`. A runner
  built around a mismatched learner accepted a carry that another runner had
  initialised.
- A learner runner (`PPORunner`, `DQNRunner`, `SACRunner`) called inside
  `jax.jit` raises an error that says it must be the outermost compiled call
  and to use a bare `jax.vmap(runner)`; it failed with JAX's message about
  `compiler_options` on a nested jit.
- `scripts/generate_env_docs.py` raises for a registered id that
  `scripts/env_content.py` has no entry for. Before, an id outside the gymnax
  and JaxMARL families was given the particle-task description.
- **Integers:** an integer setting (ticks, periods, sizes, agent counts,
  window sizes, learner configuration, `compute.advance` ticks) accepts a
  NumPy or JAX integer scalar as well as a Python int, and the error for
  anything else says what type and value it got. Booleans and floats are still
  refused. This covers the native environments, runners, learners and drift
  processes; the adapter constructors (Meta-World, EnvPool, JaxMARL, AgarCL,
  MJX, Gymnasium, PettingZoo, MORPHEUS) still need Python ints. Float drift settings (`RandomWalk.scale`, `Accumulate` gain, decay,
  bounds and bands) are stored as Python floats, so a NumPy float no longer
  breaks `checkpoint.save` after the run.
- **Drift bounds:** `RandomWalk`, `Procedural` and `Accumulate` take `low` and
  `high` as required arguments. The old `[0, 1]` defaults clipped the
  environment's starting value (CartPole gravity 9.8 became 1.0 at tick 0) and
  made `Accumulate()` a no-op on `ForagerGrid`.
- **Drift values:** a value that is not a drift process (a list, or
  `pp.drift.schedule(...)` data instead of `pp.drift.Schedule(...)`) raises a
  `TypeError` that says so, where it used to report a missing attribute of the
  environment. An environment with no driftable parameter says so, and the
  "available parameters" lists name only parameters that can drift.
- **Bare drift process:** `pp.compose(env, process)` and `env.with_drift(process)`
  raise the constructor's "needs an explicit drift mapping" error for an
  environment with several parameters and no `default_drift_field`, where they
  silently drove the alphabetically first one.
- `pp.Oracle(env, expose="task_id")` (a string, not a list) raises a
  `TypeError` that shows the fix.
- **Seeds:** `init_lifetime` (and the runners' `init`) refuse a seed outside
  `[0, 2**32)` (`2**63` with `jax_enable_x64`), where seeds that differ by a
  multiple of 2**32 gave identical lifetimes, and say what to use when given a
  PRNG key. A float seed raises the same `ValueError` (it reached JAX's own
  message before).
- A drift class passed where an instance belongs (`drift=pp.drift.Cyclic`)
  raises a `TypeError` that says to construct it. `ProbedRunner(memory=1)` on
  a single-agent environment says to raise `memory`. A checkpoint
  configuration difference is reported as `metadata.<key>...`.
- **Checkpoint errors** name what differs: the first differing configuration
  keys with both values, or the first differing container or array leaf with
  its dtype and shape. A path that does not exist raises `FileNotFoundError`.
- **Missing extras:** `pp.agents`, the gymnax, JaxMARL, MEAL and MJX ids, the
  PettingZoo adapter and `perpetua.torch_bridge` raise an `ImportError` that
  names the extra to install.
- **Regret reference:** a `compiled=True` host lift refuses `reference=` when
  the runner is built, as the docs promise, instead of failing inside the
  first compiled chunk.
- **Compute-aware time:** `ComputeRunner` and `ComputePPORunner` refuse a
  continuous-action environment when built. `compute.advance` accepts a Python
  integer tick count. The first decision's held command is action 0 (now
  documented).
- **Probes:** `dormant_unit_ratio` and `effective_rank` need at least two
  rows. A single row always read as fully dormant, rank 1.
- `pp.testing.check_env` follows the current names: `boundary`, `task_id` and
  the per-field aliases must not appear in public `info`, `boundary` (not the
  deprecated `drift_changed`) must be a declared Oracle field, a NaN in a state
  no longer reads as non-reproducible, and a state that is a registered pytree
  class no longer fails the state-type check.
- `measurement.count_compilations` no longer reports 0 when logging is
  disabled (`logging.disable`, or the JAX logger muted); it lifts those for
  the window and restores them, so `check_env`'s recompilation check cannot
  pass silently.
- **gymnax (generic) adapter:** `MetaMaze-misc`, `GaussianBandit-misc`,
  `MNISTBandit-bsuite` and `MemoryChain-bsuite` declare an unbounded
  observation box, because their observations left the box they declared
  before. Code that reads `observation_space.low` or `.high` for these four
  sees infinities now.
- `pp.testing.check_env` also checks observation dtypes, boolean masks and
  event flags, floating-point rewards and, for discrete actions, the shape
  and validity of the action mask.
- `DriftEnv` refuses drift on a field outside a world's declared
  `drift_fields`; `Schedule` checks the shape of `start`.
- Checkpoints: a host simulator that does not declare itself checkpointable
  is refused; an explicit empty `expected_metadata` is honoured.
- Config arguments such as `activation`, `torso` and `on_switch` are typed as
  `Literal`, so a type checker catches a wrong value.
- A native `TaskSwitch` over continuous tasks keeps each task's own
  `action_mask` instead of padding it with `False` entries for real action
  dimensions.
- Host workers (COOM, AgarCL): a truncated or malformed reply raises, and a
  worker that exits abnormally on `close()` warns.
- MORPHEUS client: a non-JSON or malformed response raises.
- `Regime` rejects all-zero, negative, NaN and infinite weights, and priority
  names other than `low`, `medium`, `high` and `critical`, naming the regime
  and the value; so do an out-of-range `arrival_rate` and a non-numeric
  weight. A connection failure names the base URL.
- `TorchPPO` raises when the samples do not divide into minibatches, and its
  health check covers every floating-point tensor. `TorchRunner` now raises
  this when it is built, and also for a rollout that leaves one sample per
  minibatch (the advantage normalisation turned the weights into NaN).
- **gymnax (generic) adapter actions** are checked for shape and dtype: an integer
  array of shape `(1,)` for a discrete environment, a float array of the action
  space's shape for `MountainCarContinuous`, `Reacher` and `Swimmer`. Before,
  `pp.policies.random` ran those three as a constant zero action, a scalar
  raised an `IndexError`, and a float or longer action was accepted. The error
  points to `pp.policies.uniform(env.action_space)`. Values are not checked: an
  out-of-range index reaches the backend as it is (documented, with the measured
  behaviour). Integer actions are cast to the backend's integer dtype.
- **Learners:** `PPO`, `IPPO`, `DQN` and `IQL` name the learner that was built
  and both action counts when `action_size` differs from the mask (a task
  switch pads to its largest task), refuse a continuous action space naming
  `pp.agents.SAC`, and refuse the default MLP torso on pixel observations
  naming `torso="cnn"`. The minibatch divisibility errors give the sample
  count and say whether it is pooled. `DQNConfig` refuses `rollout_steps`
  above `buffer_capacity` (the oldest ticks of each rollout were overwritten
  before any update) and documents its fields, which count ticks.
  `SAC` refuses bounds of more than one slot and says to pass
  `action_space.low[0]`.
- **Task switching:** a fixed task schedule (`Cyclic`, `Schedule`,
  `Stationary(value)`, or a `Sequence` of them) whose values are not integers
  in `[0, tasks - 1]` raises when the switch is built, where it was rounded
  and clipped silently (`Cyclic([1.0, 2.0], p)` over two tasks never
  switched but declared a boundary every `p` ticks). Random schedules are
  still rounded and clipped, now documented. A task wrapped in `Oracle` or
  `OneHotObservation` raises a `TypeError` that says to wrap the switch instead.
- **Meta-World:** `max_episode_steps` above the task's own horizon (500)
  raises, where the episode ended at 500 while the metadata recorded the larger
  value. The lifted `observation_space` declares the goal entries
  `(-inf, inf)`, because upstream declares them as the point 0 and the
  observations lie off it, and the upstream passive checker no longer prints
  three warnings.
- **Gymnasium host lifts** keep the environment's observation shape (with the
  slot axis) instead of flattening it, so a `(H, W, C)` image gives
  `(1, H, W, C)`; values are cast to float32 as before. One-dimensional
  environments are unchanged.
- Vmapping or jitting a host lift over the lifetime seed raises a `TypeError`
  that says host simulators cannot be vmapped, where JAX's tracer error
  mentioned neither Perpetua nor simulators.
- `PartnerGrid` and `OpponentGrid` check their own `size`, `num_learners` and
  scripted counts, and the error states the limit `min(32, size**2 // 2)` with
  the totals; it was phrased for `CooperativeGrid`'s `num_agents`.
- `--help` prints usage and runs nothing for `generate_env_docs.py`,
  `probe_gymnax.py`, `audit_jaxmarl.py` and `examples/custom_env.py`
  (the docs generator rewrote the docs tree on any invocation).
  `reproduce_continual_world.py metrics` exits with an error for a glob that
  matches nothing, for reference runs that did not finish healthy, and for
  reference runs whose evaluation count differs from the CL curve; it
  truncated the baseline to the shortest reference before.
- A checkpoint mismatch of 64-bit types says `jax_enable_x64` must be set the
  same way when saving and restoring.
- `GymnasiumAdapter` and `PettingZooAdapter` document that the first reset
  without a seed is seed 0 (two fresh adapters run the same lifetime), and
  that the Gymnasium view refuses reset options which the PettingZoo view
  ignores. `MetaWorldLifetime` documents that `seed` fixes the goal set.
- `scripts/measure_baseline.py` refuses a `--num-minibatches` that does not
  divide the rollout batch into minibatches of at least two samples (a
  one-sample minibatch made the loss NaN), and its progress line reports
  `null`, not `NaN`, before the first episode ends.
- `scripts/plot_claim_figures.py reset` raises on a report with more than two
  batch sizes; it drew the first two and dropped the rest.
- `GymnasiumAdapter.reset(seed=...)` and `PettingZooAdapter.reset(seed=...)`
  refuse a seed outside `[0, 2**32)` (`[0, 2**63)` with `jax_enable_x64`), as
  `init_lifetime` does; `2**32` used to start the same lifetime as 0, and the
  PettingZoo view took `-1` as `2**32 - 1`.
- `ForagerGrid(max_mass=...)` accepts any real number (`8`, `np.float32(8.0)`)
  and stores a Python float; NaN and infinity are refused, where they were
  accepted and NaN froze the agent.
- `PPOConfig`, `DQNConfig` and `SACConfig` refuse a `reward_centering` that is
  not a bool (a truthy string turned centering on) and a
  `centering_step_size` that is not finite or not in `[0, 1]`, when the config
  is built and no longer at `agent.init`.
- `SAC.init` refuses a NaN or `-inf` `target_entropy` override; it replaced
  them with the default. `+inf` still means "unset".
- A task table that names a missing task is refused on every way of binding
  it to a switch: a non-`dict` mapping, `with_drift` and `pp.compose`, as it
  already was in `pp.tasks.switch` with a bare process or a `dict`.
- `pp.tasks.switch` refuses a task whose `MultiDiscrete` action space has
  different sizes per slot when it is built, naming the task and its `nvec`;
  it failed later with a negative padding width.
- `metrics.update` refuses a `switch` that is not a scalar of boolean or
  integer dtype; a float or an array was cast to bool and could reshape the
  carry.
- `protocol.zero_shot_reward` requires `delta >= 1`.
- **`pp.testing.check_env`** steps more than the all-zero action. The shapes,
  public types and reproducibility checks step four all-zero ticks and then
  `ticks` ticks with actions sampled from a fixed key (among the actions each
  tick's mask allows; uniformly within a `Box`), and look at every tick. The
  runner-based checks (continuity, compilation, checkpoints) fail when the
  random lifetime's `metrics.healthy` is false, which a non-finite reward
  causes. An environment whose reward is NaN or infinite away from action 0,
  or whose mask closes after a move, used to pass. All 53 built-in ids that
  build on macOS pass the stricter checker unchanged.
- **`pp.tasks.switch`** refuses a `task` schedule that can leave the task
  range instead of clipping it: a `RandomWalk` or `Procedural` whose bounds
  reach outside `[-0.5, tasks - 0.5]`, and a `Sum` whose tables and bounds add
  up to such a value (`Sum([Cyclic([0.0, 7.0], 3), Stationary()])` over two
  tasks was accepted). `pp.runtime.assign_schedule` refuses a `task` schedule
  that names a missing task when its values are concrete; for a stacked batch
  assigned under `jax.vmap`, call `env.check_schedule("task", schedules)`
  first. A `RandomWalk(low=-3, high=3)` over two tasks, accepted and clipped
  before, now raises.
- **Per-task drift inside a switch** (`tasks[i].<field>`) passes task `i`'s
  own `drift_fields` and `check_drift` when the switch is built, with the
  task named in the message. Before, a protected field was accepted and
  failed at `init` with an `AttributeError`.
- **Endogenous drift** (`Accumulate`) is refused when drift is bound to an
  environment that defines no `signal`, and on every field of a task switch,
  instead of raising at the first step.
- **`gymnax/Acrobot-v1`:** `link_length_2` is no longer a driftable field.
  gymnax 0.0.9's Acrobot never reads it (only `link_length_1`), so a drift on
  it declared switches to the Oracle and to recovery tracking while the world
  stayed the same. `drift={"link_length_2": ...}` and
  `drift_fields=("link_length_2",)` raise `ValueError` with that reason; the
  generic adapter keeps a table of such fields. A test sweeps every driftable
  field of the generic gymnax ids for one that changes nothing in 300 random
  ticks, with a listed reason and a witness setting for the eight that bind
  only in states random play does not reach (Acrobot's `max_vel_*`, the
  MountainCar `max_position`, `goal_position` and `goal_velocity`).
- **`jaxmarl/MPE_simple_spread-v3`** refuses drift on `scripted`
  (`scripted_drift=` or the `drift` mapping) when `num_scripted` is 0, as the
  generic MPE adapter does; it was accepted and declared switches that changed
  nothing. Its page no longer lists `scripted` as driftable by default.
- **`pp.make`** raises `ValueError` when `partner_drift=`, `opponent_drift=`
  or `scripted_drift=` and the `drift` mapping both name the same field.
  Before, the keyword silently replaced the mapping's process.
- An environment's `check_drift` now runs before the generic `drift_fields`
  refusal, so its own message (why a field is closed) is the one raised.
- **`morpheus/Morpheus-v0`:** `base_url=`, `headers=` or a live `client=`
  with `mock=True` (the default) raise `ValueError` naming `mock=False`.
  Before, the mock ran and ignored them. With `mock=True` and `MORPHEUS_URL`
  set, the mock still runs and one warning says so.
- **`HttpMorpheusClient`** warns about a credential sent over plain `http` to
  a remote host in more cases: a header whose name contains `auth`, `token`,
  `key`, `secret` or `cookie` in any letter case (`X-Auth` and `apikey` were
  missed), `auth=` on a preconfigured `httpx` client, and user-info in the
  address. The class docstring and the MORPHEUS page list exactly what is
  covered.
- **`AECLifetime`** refuses a turn-based game at construction: an environment
  whose `metadata["is_parallelizable"]` is `False` (`connect_four_v3`,
  `tictactoe_v3`) raises `ValueError`. Before, it was accepted and failed at
  `init` with a message about action masks, although the guide said such
  environments were refused. Environments that do not declare the key are
  accepted as before.

### Documentation

- The training guide says that a learner runner must be the outermost compiled
  call, that PPO minibatches need at least two samples, and that the PPO
  example collects 128 ticks per rollout, 512 transitions across its four
  slots (it said 128 transitions). The learner options page says when the
  pixel scale is checked and that a 0-255 space of symbolic indices derives
  255 unless `pixel_scale=1` is set.
- `docs/morpheus.md` says that request headers (an access token) go out with
  every request, that a remote service should be reached over `https`, and
  how the live client bounds its record of seen tickets.
- The evaluation guide says where same-state regret reads exactly zero
  whatever the policies, and lists the environments measured: 16 where a
  tick's reward cannot depend on that tick's action (CartPole, MountainCar,
  MinAtar Breakout, UmbrellaChain and twelve MPE tasks) and 4 that read zero
  under random play only. The page on choosing an environment no longer
  recommends regret on CartPole.
- The code blocks of the API reference pages run as tests; five of them used
  names they did not define and are now complete programs.
- `CONTRIBUTING.md` lists what CI does not run (GPU-only tests, the live
  MORPHEUS test, guide blocks exempted by name, macOS). The contributing guide
  and the installation page give one test command,
  `uv run --all-extras pytest -n auto`.
- The installation page and the README say that macOS needs an arm64 Python,
  what the failure looks like otherwise and how to fix it, and that macOS is
  checked by hand.
- CI prints skip reasons (`-rs`). The torch-gated tests, including the PyTorch
  block of the training guide, and the EnvPool snapshot conformance case run
  on pull requests. The environment-page checks name the missing runtime when
  they skip, and `scripts/generate_env_docs.py --check` names the pages it
  could not build. `examples/measure_switching.py` is run by a test.
- The API reference for `Environment`, `Info`, `EnvInfo` and `Metadata` lists
  every member and key. The `action_mask` rule for continuous actions, the
  non-positive-reward case of `protocol.adaptation_speed`, the pooled median in
  `reporting.report`, the 32-bit seed range and the default drift field of the
  generic gymnax ids are stated where they apply.
- The published autoreset FLOP overheads were wrong and are corrected in
  `docs/measurement.md`.
- The measured-claims pages are corrected against their scripts and tables: the
  claim-5 adapter line counts, the claim-2 B3 counts and the reset sweep's
  environment count are recounted, the claim-4 100-configuration projection is
  replaced by the job array's linear figure, and the README quotes 41–49× for
  128 lifetimes. The Continual World pages say that `success` is read from a
  single-task `MetaWorldLifetime`, not from the switched preset. Three tutorial
  outputs (task switching, Continual World SAC, Bean World) are regenerated or
  reworded to match what the code prints.
- The Continual World preset page reports a 3-seed result with the fixed SAC
  (orthogonal and Glorot initialisation) next to the earlier 10-seed result,
  which was run before the SAC fix above.
- Environment pages have a "Default drift field" row: the parameter a bare
  `drift=process` drives. For the generic gymnax ids it is not always the first
  entry of the alphabetical "Driftable parameters" row (`reward_prob` on
  BernoulliBandit, `reward` on MetaMaze, `min_position` on MountainCar,
  `min_action` on MountainCarContinuous, `torque_scale` on Reacher), which is
  what the adapters guide said.
- The AgarCL Grand Arena page lists the defaults `pp.make` uses (128-pixel
  screen, arena 128, 500 pellets, 10 viruses, 8 bots), not the constructor's.
  The `MPE_simple_v3` page no longer offers scripted slots, which a task with
  one controllable agent rejects.
- Corrected statements: `success` is not an Oracle field of
  `continual_world(...)` in either mode (Continual World SAC tutorial);
  `Anchor.CEILING` with a negative `upper_bound` needs `alpha >= 1` and a
  positive `epsilon`; `tasks.switch` raises on a `Stationary(value)`
  task; `Accumulate` counts `elapsed` since its last band crossing;
  `assign_schedule` called mid-lifetime raises no switch event; EnvPool's
  `overlap=True` slots advance about every other tick. The installation page
  no longer says the repository is private.
- The claim-5 adapter line counts for Continual World and MORPHEUS and the
  claim-2 B3 line counts are recounted against the current sources.
- The `gymnax_params` module no longer calls its step "bit-identical" to
  gymnax's: rewards and respawns are equal on all 20 ids and observations are
  bit-equal on 18, while Acrobot's and Reacher's differ in the last float32
  bits in compiled code.
- The MetaMaze page describes its respawns: with the step limit removed an
  episode ends only when the resampled start cell is the goal (11 respawns
  and 678 goal reaches in 200,000 random ticks), and the step-count
  observation entry grows without bound between respawns.
- `ParamsEnv`'s docstring attaches the `space_change` sentence to
  `mask_params`, which it describes, instead of `drift_fields`.
- Task-switching pages no longer say tasks "run on their default parameters":
  a task's own `DriftEnv` drift is refused, but its fields may drift through
  the switch's mapping (`tasks[i].<field>`).
- Every link in `README.md` is absolute, so it works on the package index.
- The `[docs]` links on the source pages of `perpetua.envs` modules lead to
  the API reference; 88 of them pointed at anchors that do not exist on the
  environment pages.
- `CONTRIBUTING.md` gives the current sum of `.test_durations` (about 51
  minutes), and `scripts/measure_baseline.py` says how to run it from the
  project environment with the `torch` extra.

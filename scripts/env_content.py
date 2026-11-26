"""Hand-written prose for the generated environment pages.

``scripts/generate_env_docs.py`` combines this text with live metadata (spaces,
drift fields, Oracle fields, constructor arguments) into one page per
environment and one page per family. Each environment entry has the
Gymnasium-style sections adapted to lifetimes: there is no "episode end", so
"Respawns" says what happens instead. Families give shared text for the
generic gymnax and MPE pages, which have no entry of their own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import cast

import perpetua as pp


@dataclass(frozen=True)
class Family:
    slug: str
    title: str
    intro: str
    install: str
    shared: str = ""


@dataclass(frozen=True)
class Content:
    family: str
    constructor: str  # "module:Class" whose keyword arguments pp.make forwards
    description: str
    observations: str = ""
    actions: str = ""
    rewards: str = ""
    initial: str = ""
    respawns: str = ""
    drift: str = ""
    no_drift: str = ""  # replaces ``drift`` when the environment has no driftable parameter
    limits: str = ""
    guide: str = ""  # a longer page under docs/, linked from the environment page
    history: tuple[str, ...] = ()  # empty: "initial release" under the id's own version
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    title: str = ""  # page title; derived from the id when empty
    # (label, url) of the upstream documentation. When ``upstream_defines`` is
    # true that page is the reference for observations, actions and rewards,
    # and ours lists only what changes in Perpetua.
    upstream: tuple[str, str] | None = None
    upstream_defines: bool = False
    # Keyword arguments pp.make accepts beyond the constructor's, as (name, default).
    make_args: tuple[tuple[str, str], ...] = ()
    # "module:NAME" of the keyword arguments the registered id overlays on the
    # constructor's defaults; the Arguments table shows the overlaid values.
    preset: str = ""


FAMILIES: dict[str, Family] = {
    "native": Family(
        "native",
        "Native continual worlds",
        "Worlds written for Perpetua in JAX. They compile, vmap and checkpoint, and they "
        "are the fastest way to study one drift question at a time: a moving goal, "
        "changing affordances, agent-driven drift, or an open-ended world.",
        "uv sync --locked",
        "Every native world takes `drift=` (one process or a mapping of parameter names "
        "to processes). Grid sizes and agent counts are fixed per lifetime.",
    ),
    "multi-agent": Family(
        "multi-agent",
        "Multi-agent worlds",
        "Interacting worlds and multi-agent particle tasks. Every slot has its own "
        "reward and masks; partner and opponent drift change the other agents' "
        "policies, which a learner can only infer from their behaviour.",
        "uv sync --locked --extra jaxmarl   # MPE tasks; the grids need nothing extra",
        "Scripted partner and opponent slots are world participants with "
        "`active_mask=False`, so learners never train on them.",
    ),
    "control": Family(
        "control",
        "Classic and MuJoCo control",
        "Control tasks with drifting physics: gymnax's JAX control suite and MuJoCo "
        "models on MJX. Time limits are removed; terminations become respawns.",
        "uv sync --locked --extra interop   # gymnax\nuv sync --locked --extra mjx       # MJX",
        "Drift addresses physical parameters by name (for example `gravity`).",
    ),
    "gymnax": Family(
        "gymnax",
        "MinAtar, bsuite and misc (gymnax)",
        "The rest of gymnax 0.0.9 through the generic `GymnaxParameters` adapter: "
        "MinAtar games, bsuite tasks and gymnax's miscellaneous environments.",
        "uv sync --locked --extra interop",
        "The adapter removes the backend time limit, turns terminations into respawns "
        "inside the lifetime, and accepts drift on any float parameter field that no "
        "other field is precomputed from (`drift_fields=` narrows the set).",
    ),
    "benchmarks": Family(
        "benchmarks",
        "Continual benchmarks",
        "Existing continual-RL benchmarks under the lifetime API: MEAL, Continual World "
        "(Meta-World), COOM, AgarCL and MORPHEUS. None is vendored; each is an optional "
        "dependency or a separately built runtime.",
        "See each page: MEAL and Meta-World are extras; COOM and AgarCL build pinned "
        "runtimes with `python -m perpetua.adapters.coom_setup` and "
        "`python -m perpetua.adapters.agarcl_setup`.",
        "Benchmark sequences are presets (see the benchmarks page); host simulators "
        "accept `compiled=True` to run inside the compiled runners.",
    ),
    "cpu-bridge": Family(
        "cpu-bridge",
        "CPU bridge",
        "Batched C++ simulators through EnvPool (Linux only), stepped by the host or, "
        "with `compiled=True`, from inside the compiled runners.",
        "uv sync --locked --extra cpu-bridge",
        "The Agents column counts EnvPool's parallel copies of the simulator (`num_envs`): "
        "independent environments stepped together, not interacting agents. ALE is "
        "GPL-2.0 and is never vendored; it is installed only with EnvPool.",
    ),
}

MPE_RESPAWNS = (
    "The backend's episode end becomes an in-place respawn. JaxMARL ends an episode after "
    "`max_steps + 1` transitions (it checks the limit before the step counter increments), "
    "so with its default of 25 `info['respawn']` is true on ticks 26, 52, 78 and so on."
)
GRID_OBS = "float32 `(agents, 4)`: normalized row, column, goal row, goal column."
GRID_ACT = (
    "Integer `(agents,)`: stay/up/down/left/right = 0/1/2/3/4. Masked or invalid actions "
    "become stay."
)

ENVS: dict[str, Content] = {
    "ContinualGrid-v0": Content(
        "native",
        "perpetua.envs.continual_grid:ContinualGrid",
        "Collect a shared goal for +1 and respawn inside the same lifetime. The goal is "
        "the drifting parameter, so the same world can be stationary, cyclic, a random "
        "walk or a procedural stream. Agents share the goal but do not collide.",
        rewards="float32 `(agents,)`: +1 on reaching the goal.",
        initial="Agents spawn uniformly outside the goal cell.",
        respawns="An agent that collects respawns elsewhere on the same tick and "
        "`info['respawn']` reports it.",
        drift="`goal` (row, column), clipped to the grid and rounded to a cell, so drift "
        "smaller than half a cell never moves it.",
        limits="Non-interacting agents: a parallel-agent contract example, not a MARL benchmark.",
        guide="native-api.md",
        observations=GRID_OBS,
        actions=GRID_ACT,
    ),
    "DoorGrid-v0": Content(
        "native",
        "perpetua.envs.door_grid:DoorGrid",
        "Reach a fixed goal while the up/down/left/right doors open and close. The "
        "action space keeps its shape but its meaning evolves: a closed door masks and "
        "blocks the matching move.",
        rewards="float32 `(agents,)`: +1 on reaching the goal.",
        initial="Agents spawn outside the goal; all doors start open.",
        respawns="Collecting respawns the agent elsewhere on the same tick.",
        drift="`doors`, a `(4,)` vector (1 open, 0 closed). `info['space_change']` is "
        "true exactly on ticks where the effective action availability changes.",
        guide="evolving-spaces.md",
        observations=GRID_OBS,
        actions=GRID_ACT,
    ),
    "ForagerGrid-v0": Content(
        "native",
        "perpetua.envs.forager_grid:ForagerGrid",
        "Agent-driven drift: collecting adds hidden mass, mass decays, and a move "
        "succeeds with probability `1 / mass`. The drift is caused by the agent's own "
        "behaviour (pair it with `pp.drift.Accumulate`).",
        rewards="float32 `(agents,)`: +1 on reaching the fixed bottom-right goal.",
        initial="Agents start light (every move succeeds).",
        respawns="Collecting respawns the agent elsewhere on the same tick.",
        drift="`mass` per agent, fed by the environment's privileged collection signal; "
        "exposed only as the Oracle field `mass_parameters`. With `pp.drift.Accumulate`, "
        "each tick `mass = clip(decay * mass + gain * collected, low, high)`, so it "
        "stays at or below `max(low, gain / (1 - decay))` once the starting value has decayed "
        "to that bound; `low` is a floor, not a rest value.",
        observations=GRID_OBS,
        actions=GRID_ACT,
    ),
    "BeanWorld-v0": Content(
        "native",
        "perpetua.envs.bean_world:BeanWorld",
        "A never-ending foraging world in the style of Jelly Bean World (Platanios et "
        "al., 2020; a new JAX implementation, not a port). A seeded torus holds jelly "
        "beans, bananas and onions in banded biomes, so what an agent finds changes as "
        "it explores; collected items regrow after a per-type delay.",
        observations="float32 `(agents, view*view*3 + 3)` (`view` is 7 by default, so 150 "
        "values). The first `view*view*3` values are the egocentric item window, flattened "
        "row-major with the item type as the fastest axis: reshape them to "
        "`(view, view, 3)` as (row, column, item), not `(3, view, view)`. The agent is at "
        "the centre and row 0 is the top (the up action moves to the previous row). Items "
        "are ordered jelly bean, banana, onion. The last three values are the per-item scent "
        "in the same order: the sum over the `(2*scent_radius+1)`-square around the agent of "
        "present items weighted by `1/(1+Manhattan distance)`. It is not normalised to "
        "[0, 1]; at the default `scent_radius` of 8 it is at most about 37.6 per item.",
        actions="Integer `(agents,)`: stay/up/down/left/right, always valid on the torus.",
        rewards="float32 `(agents,)`: the item's reward when stepping onto one "
        "(defaults: jelly bean +1, banana +0.1, onion -1).",
        initial="A seeded layout; agents spawn uniformly.",
        respawns="None: agents never die or respawn.",
        drift="`item_rewards` and `regrow_ticks` (per item type), hidden unless an Oracle "
        "exposes them.",
        limits="Agents that step onto the same item on the same tick each receive it.",
    ),
    "CooperativeGrid-v0": Content(
        "multi-agent",
        "perpetua.envs.cooperative_grid:CooperativeGrid",
        "A cooperative grid with real interaction: agents block one another, contend "
        "for free cells, and the whole team scores only when every active agent "
        "reaches its target.",
        observations="float32 `(agents, 4 + 3*(agents-1))`: own position and target, "
        "then each other slot's position and activity.",
        actions=GRID_ACT,
        rewards="float32 `(agents,)`: +1 each when all active agents reach their targets.",
        initial="Unique spawns outside the targets.",
        respawns="After a joint collection every agent respawns uniquely.",
        drift="`goal`, the anchor of the consecutive row-major targets.",
        guide="marl.md",
    ),
    "PartnerGrid-v0": Content(
        "multi-agent",
        "perpetua.envs.partner_grid:PartnerGrid",
        "Cooperate with scripted partners whose cooperativeness drifts. Each partner "
        "moves beta-greedily toward its target or at random; beta is hidden.",
        observations="As `CooperativeGrid-v0`; partner slots observe zeros.",
        actions=GRID_ACT + " Partner slots allow only stay.",
        rewards="float32 `(agents,)`: the joint +1 when every slot reaches its target.",
        initial="Unique spawns outside the targets.",
        respawns="Everyone respawns after a joint collection.",
        drift="`goal`, and `partner` (each partner's beta in [0, 1]): pass `partner_drift=` "
        'or a mapping such as `drift={"goal": ..., "partner": ...}`.',
        make_args=(("partner_drift", "None"),),
        limits="`size` is at most 64, and `num_learners + num_partners` is at most "
        "`min(32, size**2 // 2)` slots.",
        guide="marl.md",
    ),
    "OpponentGrid-v0": Content(
        "multi-agent",
        "perpetua.envs.opponent_grid:OpponentGrid",
        "Learners collect a shared goal while scripted opponents with drifting "
        "aggression hunt them. Rewards are zero-sum.",
        observations="As `CooperativeGrid-v0`; opponent slots observe zeros.",
        actions=GRID_ACT + " Opponent slots allow only stay.",
        rewards="float32 `(agents,)`: +1 for collecting, -1 when caught; opponents share "
        "the negated learner sum. A learner is caught when, after the tick's moves, it is "
        "within one step (Manhattan distance 1 or less, so the same cell counts) of any "
        "opponent; a catch cancels a collection on that tick.",
        initial="Learners spawn uniformly outside the goal cell, drawn independently, so two "
        "learners can share a cell. Opponents spawn uniformly over every cell without "
        "replacement, so two opponents never share a cell and one can start on a learner or "
        "on the goal. When only opponents propose the same free cell, the lowest slot among "
        "them moves and the others stay; a cell a learner also proposes stays empty. On the "
        "default 5-by-5 grid, 3.75% of 2,000 seeds "
        "start with the learner and the opponent on one cell and 16.8% start within catch "
        "range.",
        respawns="A caught or collecting learner respawns outside the goal, with no regard to "
        "the opponents, so it can be caught again on the next tick.",
        drift="`goal`, and `opponent` (each opponent's aggression in [0, 1]): pass "
        '`opponent_drift=` or a mapping such as `drift={"goal": ..., "opponent": ...}`.',
        make_args=(("opponent_drift", "None"),),
        limits="`size` is at most 64, and `num_learners + num_opponents` is at most "
        "`min(32, size**2 // 2)` slots.",
        guide="marl.md",
    ),
    "gymnax/CartPole-v1": Content(
        "control",
        "perpetua.adapters.gymnax:GymnaxCartPole",
        "gymnax CartPole as a continuing lifetime with drifting gravity.",
        rewards="float32 `(1,)`: +1 per surviving tick; the failure tick scores zero.",
        respawns="A failure respawns the pole in place on the same tick.",
        drift="`gravity` (m/s², shape `(1,)`): write phases as tuples, e.g. "
        "`Cyclic([(9.8,), (12.0,)], 1000)`.",
        guide="adapters.md",
    ),
    "gymnax/Pendulum-v1": Content(
        "control",
        "perpetua.adapters.gymnax:GymnaxPendulum",
        "gymnax Pendulum with continuous torque and drifting gravity.",
        rewards="float32 `(1,)`: Gymnasium's pendulum cost, negated.",
        respawns="None: Pendulum has no failure states.",
        drift="`g` (gravity).",
        guide="adapters.md",
    ),
    "mjx/Pendulum-v0": Content(
        "control",
        "perpetua.adapters.mjx:MJXControl",
        "A swing-up pendulum on MuJoCo MJX, from an MJCF model written for Perpetua. "
        "It compiles, vmaps and checkpoints like a native environment.",
        observations="float32 `(1, 3)`: cos and sin of the angle from upright, angular velocity.",
        actions="float32 `(1, 1)`: torque in [-2, 2].",
        rewards="float32 `(1,)`: `-(angle**2 + 0.1*velocity**2 + 0.001*torque**2)`.",
        initial="Hanging down with a small seeded perturbation.",
        respawns="None: no termination or time limit.",
        drift="`gravity` (magnitude) and `mass_scale` (bodies' mass and inertia, with the "
        "constants MuJoCo derives from them, so the model equals one rebuilt at that mass).",
        extra_args=("task",),
    ),
    "mjx/CartPole-v0": Content(
        "control",
        "perpetua.adapters.mjx:MJXControl",
        "A swing-up cart-pole on MuJoCo MJX. The cart is held by joint limits, so the "
        "pole can always be swung up again.",
        observations="float32 `(1, 5)`: cart position and velocity, cos and sin of the "
        "pole angle, angular velocity.",
        actions="float32 `(1, 1)`: normalized force in [-1, 1].",
        rewards="float32 `(1,)`: `cos(angle) - 0.01*position**2 - 0.001*force**2`.",
        initial="Hanging down with a small seeded perturbation.",
        respawns="None: no termination or time limit.",
        drift="`gravity` and `mass_scale`, as in `mjx/Pendulum-v0`. The slider limit of the "
        "cart responds to the scaled mass too.",
        extra_args=("task",),
    ),
    "jaxmarl/MPE_simple_spread-v3": Content(
        "multi-agent",
        "perpetua.adapters.jaxmarl:JaxMARLSimpleSpread",
        "JaxMARL's cooperative navigation as a continuing lifetime, with optional "
        "scripted partner or opponent slots whose cooperativeness drifts.",
        respawns=MPE_RESPAWNS,
        drift="`reward_multiplier`, the dynamics fields listed under Driftable parameters, "
        "and, with `num_scripted=` at least 1, `scripted` (cooperativeness in [0, 1]) via "
        "`scripted_drift=`. Without scripted slots `scripted` is not driftable.",
        guide="marl.md",
    ),
    "meal/overcooked-v0": Content(
        "benchmarks",
        "perpetua.adapters.meal:MealOvercooked",
        "One MEAL Overcooked layout (MIT, MEAL 0.3.1), generated through MEAL's seeded "
        "sequence API, as a continuing lifetime. `pp.presets.meal_sequence` cycles "
        "layouts.",
        observations="float32 `(agents, height, width, 26)`, channels last, one grid per cook "
        "with that cook's own layers first. As MEAL's `get_obs` documents them: channels 0-9 "
        "are own and partner position, then own and partner facing (four channels each, one per "
        "direction); 10-15 the layout (pots, counters, onion piles, tomato piles that this "
        "environment never uses, plate piles, delivery squares); 16-21 the pot state (onions "
        "waiting, tomatoes, onions cooking, tomatoes, cooking time left, soup done); 22-24 "
        "loose plates, onions and tomatoes; 25 an urgency flag that is 1 on every cell when 40 "
        "or fewer steps remain.",
        actions="`Discrete(6)` per cook, as MEAL's `Actions`: 0 up, 1 down, 2 right, 3 left, "
        "4 stay, 5 interact.",
        rewards="float32 `(agents,)`: MEAL's sparse reward, 20 to the cook that delivers a "
        "soup. MEAL's shaped reward is not exposed.",
        respawns="MEAL's auto-resets become respawns inside the lifetime.",
        drift="`reward_multiplier`; layout switching is a preset.",
        guide="benchmarks.md",
    ),
    "metaworld/MT1-v0": Content(
        "benchmarks",
        "perpetua.adapters.host:MetaWorldLifetime",
        "One Farama Meta-World 3.1.1 MT1 task on MuJoCo 3.3.0, host-controlled. "
        "`seed` fixes the 50 goal and object configurations MT1 generates and "
        "`task_index` pins one of them, so lifetimes with different `seed` values "
        "face different goals for the same `task_index`. `pp.presets.cw10()`/`cw20()` "
        "give the Continual World order.",
        respawns="Backend episodes (200 steps by default) end in respawns.",
        drift="`reward_multiplier`; task switching is a preset.",
        limits="Opaque simulator state: not checkpointable. `max_episode_steps` may not "
        "exceed Meta-World's own horizon of 500 steps. Upstream declares the goal "
        "entries of the observation (36 to 38) as a single point; the lifted space "
        "declares them `(-inf, inf)`.",
        guide="benchmarks.md",
    ),
    "coom/COOM-v0": Content(
        "benchmarks",
        "perpetua.adapters.coom:COOMLifetime",
        "One COOM (MIT) ViZDoom scenario and task, run as a worker in COOM's own pinned "
        "runtime. `pp.presets.coom_sequence('CO8')` cycles a COOM sequence.",
        observations="float32 `(agents, 84, 84, 12)`: COOM's default wrappers (resize to "
        "84x84, rescale, running normalisation) with the four stacked RGB frames merged "
        "into channels.",
        actions="`Discrete(12)` per agent, COOM's unified action space "
        "(`build_multi_discrete_actions`): turn (none, right, left) by move forward (no, "
        "yes) by execute (no, yes), with index `4*turn + 2*move + execute`. What execute "
        "does depends on the scenario: it presses a button, shoots, jumps or accelerates.",
        rewards="float32 `(agents,)`: the scenario's own reward, unchanged. See "
        "[COOM](https://github.com/TTomilin/COOM) for each scenario.",
        respawns="Scenario episode ends become respawns.",
        drift="`reward_multiplier`; sequences are presets.",
        limits="Opaque simulator state: not checkpointable. "
        "Needs `python -m perpetua.adapters.coom_setup`.",
        guide="benchmarks.md",
    ),
    "morpheus/Morpheus-v0": Content(
        "benchmarks",
        "perpetua.adapters.morpheus:MorpheusLifetime",
        "A persistent MORPHEUS enterprise world: incident tickets raised by a chaos "
        "engine, remediated and verified. The default is a seeded mock; `mock=False` "
        "and `MORPHEUS_URL` use a live deployment over HTTP.",
        observations="Thirteen values: open tickets; open tickets by priority (low, medium, "
        "high, critical) and by status (new, open, in_progress, on_hold); the "
        "verifier pass fraction of this tick's resolve (0 on a tick without one); "
        "in-transit shipments; units processed since the lifetime began; elapsed "
        "simulated hours.",
        actions="`Discrete(5)`: 0 no-op, 1 triage the highest-priority untriaged ticket, "
        "2 resolve and verify the highest-priority in-progress (else open) ticket, "
        "3 order 25 units for inventory, 4 deliver the oldest in-transit shipment. "
        "Actions 3 and 4 land two simulated hours after the action, in the result of the "
        "following tick; the delivered shipment leaves transit at once and its unit is "
        "counted two hours later. An action without a target is a no-op.",
        rewards="`reward='paper'` (default) is the paper's composite, "
        "`0.5·r_failure + 0.25·r_ledger + 0.25·r_throughput`, computed per tick. "
        "`'failure'`, `'verification'`, `'capital'` or a callable select others; see "
        "the guide for the definitions and how this differs from the paper's schedule.",
        respawns="None: the world keeps running.",
        drift="`regime`, pushed to the world at each switch; Oracle-only "
        "(`regime_index`, `task_id`, `chaos_active`, `invalid_action`).",
        limits="Not checkpointable (the live clock lives on the service).",
        guide="morpheus.md",
    ),
    "agarcl/agario-screen-v0": Content(
        "benchmarks",
        "perpetua.adapters.agarcl:AgarCLLifetime",
        "AgarCL (MIT) with RGB screen observations, in its own pinned native runtime. "
        "Actions are `Box(3)`: x/y movement plus noop/split/feed.",
        observations="float32 `(agents, screen_len, screen_len, 3)`: the RGB screen, values "
        "0 to 255, `(1, 84, 84, 3)` by default; `flatten=True` flattens it.",
        actions="float32 `(1, 3)`: x and y movement in [-1, 1], then a value the adapter "
        "rounds and clips to 0 (no-op), 1 (split) or 2 (feed). The Action Values row shows "
        "the overall range `[-1, 2]`.",
        rewards="float32 `(agents,)`: the change in the agent's mass over the tick, "
        "AgarCL's default reward.",
        respawns="Death respawns the agent; `respawn_after=` adds a periodic respawn.",
        drift="`reward_multiplier`; AgarCL's own dynamics change with the agent's mass.",
        limits="Checkpointable only with `replay_capacity` (seeded action-log replay).",
        guide="agarcl.md",
    ),
    "agarcl/agario-grid-v0": Content(
        "benchmarks",
        "perpetua.adapters.agarcl:AgarCLLifetime",
        "AgarCL with its 8-channel grid observation (bypassing upstream's broken grid "
        "wrapper). Actions are `Box(3)`: x/y movement plus noop/split/feed.",
        observations="float32 `(agents, grid_size, grid_size, 8)`, `(1, 128, 128, 8)` by "
        "default. The channels are out-of-bounds, pellets (presence and count), viruses "
        "(presence and count), the agent's own cells, and other players' minimum and maximum "
        "mass.",
        actions="float32 `(1, 3)`: x and y movement in [-1, 1], then a value the adapter "
        "rounds and clips to 0 (no-op), 1 (split) or 2 (feed). The Action Values row shows "
        "the overall range `[-1, 2]`.",
        rewards="float32 `(agents,)`: the change in the agent's mass over the tick, "
        "AgarCL's default reward.",
        respawns="Death respawns the agent; `respawn_after=` adds a periodic respawn.",
        drift="`reward_multiplier`.",
        limits="Checkpointable only with `replay_capacity`.",
        guide="agarcl.md",
    ),
    "agarcl/grand-arena-v0": Content(
        "benchmarks",
        "perpetua.adapters.agarcl:AgarCLLifetime",
        "AgarCL's full game, the Grand Arena, with the world from the paper's Appendix "
        "G.1: 500 pellets, 10 viruses, 8 heuristic bots and 128x128 screens. Every "
        "setting is a keyword override.",
        observations="float32 `(agents, 128, 128, 3)`: upstream's RGB screen, values 0 to 255.",
        actions="float32 `(1, 3)`: x and y movement in [-1, 1], then a value the adapter "
        "rounds and clips to 0 (no-op), 1 (split) or 2 (feed). The Action Values row shows "
        "the overall range `[-1, 2]`.",
        rewards="float32 `(agents,)`: the change in the agent's mass over the tick, "
        "AgarCL's default reward.",
        respawns="Death respawns the agent.",
        drift="`reward_multiplier`.",
        limits="Upstream's RGB render rather than the paper's four-channel tensor; the "
        "benchmark repository's committed config differs (arena 300, 600 pellets, 4 bots).",
        guide="agarcl.md",
        preset="perpetua.adapters.agarcl:GRAND_ARENA",
    ),
    "envpool/CartPole-v1": Content(
        "cpu-bridge",
        "perpetua.adapters.host:EnvPoolCartPole",
        "EnvPool's batched C++ CartPole, one lifetime over `num_envs` slots.",
        respawns="EnvPool's next-step autoreset becomes a respawn event.",
        drift="`reward_multiplier`.",
        limits="Linux only. `snapshot=True` needs the optional native binding; "
        "`overlap=True` steps half batches (not bit-reproducible).",
        guide="adapters.md",
    ),
    "envpool/Atari-v0": Content(
        "cpu-bridge",
        "perpetua.adapters.host:EnvPoolAtari",
        "One Atari game through EnvPool with `full_action_space=True` (18 actions), so "
        "games can switch without changing shapes (`pp.presets.switching_atari`).",
        observations="float32 `(envs, 84, 84, 4)`, channel-last, values 0 to 255: four stacked "
        "single-channel 84x84 frames per slot (EnvPool's preprocessing), not ALE's raw "
        "screen, which ALE's page describes. A PPO or DQN learner with the CNN torso, built "
        "with `observation_space=env.observation_space`, divides them by 255.",
        actions="`Discrete(18)` per slot: ALE's full action set for every game, not the "
        "game's own minimal set.",
        rewards="Each game's own score change, as defined by the Arcade Learning "
        "Environment page above; the adapter applies no rescaling beyond `reward_multiplier` "
        "drift.",
        respawns="A slot whose game ends is inactive on that tick, because EnvPool ignores "
        "the action it receives next, and its respawn is reported on the following tick "
        "with the reset frame.",
        drift="`reward_multiplier`; game switching is a preset.",
        limits="Linux and EnvPool only; ALE is GPL-2.0 and never vendored.",
        guide="adapters.md",
    ),
}


GYMNASIUM = "https://gymnasium.farama.org/environments/classic_control"
MPE2 = "https://mpe2.farama.org/environments"
UPSTREAM: dict[str, tuple[str, str]] = {
    "gymnax/CartPole-v1": ("Gymnasium CartPole", f"{GYMNASIUM}/cart_pole/"),
    "gymnax/Pendulum-v1": ("Gymnasium Pendulum", f"{GYMNASIUM}/pendulum/"),
    "gymnax/Acrobot-v1": ("Gymnasium Acrobot", f"{GYMNASIUM}/acrobot/"),
    "gymnax/MountainCar-v0": ("Gymnasium Mountain Car", f"{GYMNASIUM}/mountain_car/"),
    "gymnax/MountainCarContinuous-v0": (
        "Gymnasium Continuous Mountain Car",
        f"{GYMNASIUM}/mountain_car_continuous/",
    ),
    "envpool/CartPole-v1": ("Gymnasium CartPole", f"{GYMNASIUM}/cart_pole/"),
    "envpool/Atari-v0": ("the Arcade Learning Environment", "https://ale.farama.org/environments/"),
    "metaworld/MT1-v0": (
        "Meta-World",
        "https://metaworld.farama.org/benchmark/benchmark_descriptions/",
    ),
}
# Pages whose upstream documents the environment but not everything this page shows.
UPSTREAM_PARTIAL = {"envpool/Atari-v0", "jaxmarl/MPE_simple_crypto_v3"}
# MPE tasks that PettingZoo documented and that now live in Farama's MPE2.
MPE2_TASKS = {
    "simple": "simple",
    "simple_adversary": "simple_adversary",
    "simple_crypto": "simple_crypto",
    "simple_push": "simple_push",
    "simple_reference": "simple_reference",
    "simple_speaker_listener": "simple_speaker_listener",
    "simple_spread": "simple_spread",
    "simple_tag": "simple_tag",
    "simple_world_comm": "simple_world_comm",
}
REPOSITORIES = {
    "MinAtar": ("MinAtar", "https://github.com/kenjyoung/MinAtar"),
    "bsuite": ("bsuite", "https://github.com/google-deepmind/bsuite"),
    "misc": ("gymnax", "https://github.com/RobertTLange/gymnax"),
    "facmac": (
        "JaxMARL's MPE",
        "https://github.com/bold-lab-ai/JaxMARL/tree/main/jaxmarl/environments/mpe",
    ),
    "meal": ("MEAL", "https://github.com/TTomilin/MEAL"),
    "coom": ("COOM", "https://github.com/TTomilin/COOM"),
    "agarcl": ("AgarCL", "https://github.com/AgarCL/AgarCL"),
    "morpheus": ("MORPHEUS (paper)", "https://openreview.net/forum?id=31P1VAfLkJ"),
}
TITLES = {
    "meal/overcooked-v0": "MEAL Overcooked",
    "metaworld/MT1-v0": "Meta-World MT1",
    "coom/COOM-v0": "COOM",
    "morpheus/Morpheus-v0": "MORPHEUS",
    "agarcl/agario-screen-v0": "AgarCL (screen)",
    "agarcl/agario-grid-v0": "AgarCL (grid)",
    "agarcl/grand-arena-v0": "AgarCL Grand Arena",
    "envpool/CartPole-v1": "EnvPool CartPole",
    "envpool/Atari-v0": "EnvPool Atari",
    "mjx/Pendulum-v0": "MJX Pendulum",
    "mjx/CartPole-v0": "MJX CartPole",
    "gymnax/MountainCarContinuous-v0": "Mountain Car Continuous",
    "gymnax/MountainCar-v0": "Mountain Car",
}
# What the base install gives for an id whose extra only some options need.
OPTIONAL_EXTRA_NOTES = {"morpheus": "the mock; add --extra {extra} for a live service"}
PLATFORM_NAMES = {"linux": "Linux"}


def title_of(env_id: str) -> str:
    """A human page title, as PettingZoo's pages have (``Connect Four``)."""
    content = ENVS.get(env_id)
    if content is not None and content.title:
        return content.title
    if env_id in TITLES:
        return TITLES[env_id]
    name = env_id.split("/", 1)[-1]
    if env_id.startswith("jaxmarl/"):
        task = name.removeprefix("MPE_").rsplit("-", 1)[0]
        task = re.sub(r"_v\d+$", "", task)
        words = [
            word.upper() if word == "facmac" else word.capitalize() for word in task.split("_")
        ]
        return " ".join(words)
    if env_id.startswith("gymnax/"):
        base, _, suite = name.partition("-")
        if suite in {"MinAtar", "bsuite", "misc"}:
            return f"{base} ({suite})"
        return base
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name.rsplit("-", 1)[0])


def upstream_of(env_id: str) -> tuple[tuple[str, str] | None, bool]:
    """(label, url) of an environment's upstream docs, and whether they define it."""
    if env_id in UPSTREAM:
        return UPSTREAM[env_id], env_id not in UPSTREAM_PARTIAL
    name = env_id.split("/", 1)[-1]
    if env_id.startswith("jaxmarl/"):
        task = re.sub(r"_v\d+$", "", name.removeprefix("MPE_").rsplit("-", 1)[0])
        if task in MPE2_TASKS:
            return (
                (f"MPE2 {title_of(env_id)}", f"{MPE2}/{MPE2_TASKS[task]}/"),
                env_id not in UPSTREAM_PARTIAL,
            )
        return REPOSITORIES["facmac"], False
    if env_id.startswith("gymnax/"):
        suite = name.partition("-")[2]
        if suite in REPOSITORIES:
            return REPOSITORIES[suite], suite != "misc"
    for prefix in ("meal", "coom", "agarcl", "morpheus"):
        if env_id.startswith(f"{prefix}/"):
            return REPOSITORIES[prefix], False
    return None, False


def version_of(env_id: str) -> str:
    """The id's version suffix (``v0`` in ``ContinualGrid-v0``, ``v3`` in ``MPE_simple_v3``)."""
    match = re.search(r"[-_](v\d+)$", env_id)
    return match.group(1) if match else "v0"


def install_of(env_id: str) -> str:
    """The install command, from the registry's record of what the id needs."""
    record = pp.spec(env_id)
    if record.setup is not None:
        return f"uv run python -m {record.setup}"
    command = "uv sync --locked"
    notes: list[str] = []
    if record.extra is not None:
        command += f" --extra {record.extra}"
    if record.optional_extra is not None:
        note = OPTIONAL_EXTRA_NOTES[cast(str, record.namespace)]
        notes.append(note.format(extra=record.optional_extra))
    if record.platform is not None:
        notes.append(PLATFORM_NAMES[record.platform])
    return command + (f"  # {'; '.join(notes)}" if notes else "")


def family_of(env_id: str) -> str:
    """The family of a registered environment id."""
    return content_for(env_id).family


# (observations, actions, rewards) for the gymnax pages whose upstream is a repository root;
# each line was read from gymnax 0.0.9's source and checked on the running adapter.
BANDIT_TIME = "the step count since the last respawn, scaled to [-1, 1] over the 100-step limit"
GYMNAX_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "BernoulliBandit-misc": (
        "float32 `(1, 4)`: the last reward (0 or 1), a one-hot of the last action (2 values) "
        f"and {BANDIT_TIME}.",
        "`Discrete(2)`: the arm to pull.",
        "float32 `(1,)`: 1 with the pulled arm's probability, else 0. The arms pay with "
        "probabilities `reward_prob` (0.1) and `1 - reward_prob`, in an order drawn at each "
        "respawn.",
    ),
    "GaussianBandit-misc": (
        "float32 `(1, 4)`: the last reward, a one-hot of the last action (2 values) and "
        f"{BANDIT_TIME}.",
        "`Discrete(2)`: the arm to pull.",
        "float32 `(1,)`: arm 0 pays 0. Arm 1 pays `mu` plus Gaussian noise with standard "
        "deviation `sigma_l` (0.1); `mu` is drawn at each respawn from a Gaussian with mean "
        "`mean_mu` (-1) and standard deviation `sigma_p` (1).",
    ),
    "FourRooms-misc": (
        "float32 `(1, 4)`: the agent's row and column, then the goal's row and column, on a "
        "13-by-13 map (values 1 to 11). The agent starts at (4, 1) and the goal is at (8, 9) "
        "unless the backend's `resample_*` parameters are changed.",
        "`Discrete(4)`: 0 up, 1 right, 2 down, 3 left. With probability `fail_prob * 4/3` "
        "(4/9) a uniformly random action replaces the chosen one; a wall blocks the move.",
        "float32 `(1,)`: 1 when the agent moves onto the goal, else 0.",
    ),
    "MetaMaze-misc": (
        "float32 `(1, 15)`: the 3-by-3 window around the agent (1 for a wall), flattened "
        "row-major (9 values), a one-hot of the last action (4), the last reward and the "
        "unnormalised step count since the respawn.",
        "`Discrete(4)`: 0 up, 1 right, 2 down, 3 left; a wall blocks the move.",
        "float32 `(1,)`: `reward` (10) when the agent reaches the goal, else 0 (the "
        "backend's wall `punishment` is 0).",
    ),
    "Pong-misc": (
        "float32 `(1, 30, 40, 3)`: rows by columns by three channels. Channel 0 marks the "
        "two paddles (five cells each, in the first and last column), channel 1 the ball and "
        "channel 2 the ball's previous position.",
        "`Discrete(3)`: 0 stay, 1 up, 2 down.",
        "float32 `(1,)`: 1 on every tick the ball stays in play, 0 on the tick it leaves the "
        "court at either end.",
    ),
    "Reacher-misc": (
        "float32 `(1, 8)`: the cosines of the two joint angles, their sines, the two joint "
        "velocities, then the goal's x and y.",
        "float32 `(1, 2)` in [-1, 1]: the two joint accelerations, multiplied by `torque_scale`.",
        "float32 `(1,)`: `exp(-d**2)`, where d is the distance from the arm's tip to the "
        "goal, so in (0, 1].",
    ),
    "Swimmer-misc": (
        "float32 `(1, 16)`: the five urchins' x and y (10 values), the swimmer's position "
        "(2), its velocity (2) and the goal (2).",
        "float32 `(1, 2)` in [-1, 1]: the swimmer's x and y acceleration.",
        "float32 `(1,)`: `exp(-0.5 * d**2)` for the distance d to the goal, times "
        "`1 - exp(-30 * s**2)` for every urchin at distance s, so it falls toward 0 over an "
        "urchin.",
    ),
}
# Respawn text for ids whose termination differs from "the backend's terminations become
# respawns"; the tick numbers were measured with random policies.
GYMNAX_RESPAWNS = {
    "MNISTBandit-bsuite": "The backend ends the episode after every step, so `info['respawn']` "
    "is true on every tick.",
    "Freeway-MinAtar": "None: the backend ends an episode only at its time limit, which is "
    "removed, so there is never a respawn.",
    "Reacher-misc": "None: the backend ends an episode only at its time limit, which is "
    "removed, so there is never a respawn.",
    "Swimmer-misc": "None: the backend ends an episode only at its time limit, which is "
    "removed, so there is never a respawn.",
    "MetaMaze-misc": "Rare. Reaching the goal pays the reward and moves the agent to a "
    "resampled start cell without ending the episode. With the 200-step limit removed, an "
    "episode ends only when that resampled start cell is the goal itself, and only then is "
    "`info['respawn']` true and a new goal drawn: 11 respawns and 678 goal reaches in "
    "200,000 random ticks. The last observation entry is the backend's step count since "
    "the last respawn, so it grows without bound between respawns (1,000 at tick 1,000).",
}
FACMAC_OBSERVATIONS = (
    "float32 `(agents, d)` with d = 16, 28 or 40 for 3, 6 or 9 predators: the agent's own "
    "velocity (2), its own position (2), the landmarks' positions relative to it (2 per "
    "landmark), the other agents' relative positions (2 per other agent) and 2 values for "
    "another agent's velocity (zeros for a prey). JaxMARL zeroes the landmark and "
    "other-agent blocks when the block's norm exceeds the view radius of 1.5."
)
FACMAC_ACTIONS = (
    "float32 `(agents, 5)` in [0, 1], as MPE's continuous encoding: value 0 is unused, the "
    "x force is `a[2] - a[1]` and the y force `a[4] - a[3]`, multiplied by the slot's "
    "acceleration (3.0 for a predator, 4.0 for a prey)."
)
FACMAC_REWARDS = (
    "float32 `(agents,)`: every predator receives the same reward, +10 for each "
    "predator-prey collision on the tick. A prey receives -10 per collision minus a penalty "
    "for nearing the map's edge."
)
CRYPTO_OBSERVATIONS = (
    "float32 `(agents, 8)`, zero-padded to one width. The structure is that of "
    f"[MPE2 Simple Crypto]({MPE2}/simple_crypto/), but JaxMARL's goal and key colours use 255 "
    "where MPE2 uses 1. Alice's observation is the goal colour and the key colour, each a "
    "one-hot over four colours scaled by 255, so a row such as `[0, 255, 0, 0, 0, 255, 0, 0]` "
    "(the colours are random per reset) is `[0, 1, 0, 0, 0, 1, 0, 0]` in MPE2."
)
CRYPTO_ACTIONS = f"`Discrete(4)` per slot, as in [MPE2 Simple Crypto]({MPE2}/simple_crypto/)."
CRYPTO_REWARDS = (
    "float32 `(agents,)`, ordered Eve, Alice, Bob. The structure is that of "
    f"[MPE2 Simple Crypto]({MPE2}/simple_crypto/), but the 255-valued colours put the "
    "reward on a different scale: with a random policy Eve's reward each tick is -65026 or "
    "-64516 (the usage snippet below gives -65026 for its first four ticks), where MPE2 "
    "gives values between -2 and 2. The values are JaxMARL's own; `reward_multiplier` drift "
    "multiplies them as they are."
)
KEPT_LIMIT = {"BernoulliBandit-misc", "GaussianBandit-misc", "DiscountingChain-bsuite"}
NO_DRIFT = (
    "None: this environment has no float parameter, so `drift=` is rejected. To change the "
    "task over time, switch between environments of the same shapes with `pp.tasks.switch`, "
    "as `pp.presets.switching_minatar` does for MinAtar games."
)


def generic(env_id: str, solo: bool = False) -> Content:
    """Content for gymnax and MPE ids that share their family's adapter.

    ``solo`` marks an MPE task with one controllable agent, which rejects
    scripted slots, so its page does not offer them.
    """
    if not env_id.startswith(("gymnax/", "jaxmarl/")):
        raise KeyError(f"scripts/env_content.py has no ENVS entry for {env_id!r}.")
    name = env_id.split("/", 1)[1]
    if env_id.startswith("gymnax/"):
        control = name in {"Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0"}
        definition = GYMNAX_DEFINITIONS.get(name, ("", "", "The backend's reward, unchanged."))
        return Content(
            "control" if control else "gymnax",
            "perpetua.adapters.gymnax_params:GymnaxParameters",
            f"gymnax's `{name}` through the generic `GymnaxParameters` adapter.",
            observations=definition[0],
            actions=definition[1],
            rewards=definition[2],
            respawns=(
                "The environment has no termination other than its time limit, so the limit "
                "is kept and reported as a respawn: `max_steps_in_episode` is 100, so "
                "`info['respawn']` is true on ticks 100, 200, 300 and so on. Parameters it "
                "reads only at reset take effect at the next respawn."
                if name in KEPT_LIMIT
                else GYMNAX_RESPAWNS.get(
                    name,
                    "The backend time limit is removed; terminations become respawns "
                    "inside the lifetime.",
                )
            ),
            drift="Any float parameter listed under Driftable parameters, by name.",
            limits=(
                "Actions must be float arrays of the action space's shape, so "
                "`pp.policies.random` does not apply: sample with "
                "`pp.policies.uniform(env.action_space)`."
                if name in {"MountainCarContinuous-v0", "Reacher-misc", "Swimmer-misc"}
                else "Actions are checked for shape and dtype (an integer array of shape "
                "`(1,)`), not for value: an index outside the action range reaches the "
                "backend as it is. Most backends look the index up in a table, so an index "
                "past the end acts as the last action and a negative one counts from the "
                "end; MountainCar scales the push with it. Clip in the policy."
            ),
            no_drift=NO_DRIFT,
            guide="adapters.md",
            history=(
                f"`{version_of(env_id)}`: wraps gymnax 0.0.9; the version follows gymnax's id.",
            ),
        )
    facmac = "facmac" in name
    crypto = name.startswith("MPE_simple_crypto")
    return Content(
        "multi-agent",
        "perpetua.adapters.jaxmarl_mpe:JaxMARLMPE",
        f"JaxMARL's `{name}` through the generic `JaxMARLMPE` adapter, padded to one "
        "observation and action shape across agents"
        + (
            ". It has one controllable agent, so it takes no scripted slots."
            if solo
            else ", with optional scripted partner or opponent slots."
        )
        + (
            " FACMAC is the predator-prey task of Peng et al.'s factored multi-agent "
            "policy-gradient paper (2021): predators chase prey that move by a built-in "
            "policy. `3a`, `6a` and `9a` mean 3, 6 and 9 predators, with 1, 2 and 3 prey and "
            "2, 4 and 6 landmarks."
            if facmac
            else ""
        ),
        observations=(FACMAC_OBSERVATIONS if facmac else CRYPTO_OBSERVATIONS if crypto else ""),
        actions=FACMAC_ACTIONS if facmac else CRYPTO_ACTIONS if crypto else "",
        rewards=(
            FACMAC_REWARDS
            if facmac
            else CRYPTO_REWARDS
            if crypto
            else "The backend's per-agent reward."
        ),
        respawns=MPE_RESPAWNS,
        drift=(
            "`reward_multiplier` and the dynamics fields listed under Driftable parameters."
            if solo
            else "`reward_multiplier`, the dynamics fields listed under Driftable parameters, "
            "and `scripted` (how often each scripted slot acts on purpose, in [0, 1]) via "
            "`scripted_drift=` with "
            "`num_scripted=` or `scripted_slots=`."
        ),
        limits=(
            "The prey are moved by FACMAC's built-in policy and ignore actions, so their "
            'slots are always inactive. They are the "scripted" slots in the Agents row, '
            "which are not the optional `num_scripted` slots below. "
            + (
                "This id and `MPE_simple_facmac_3a_v1` have the same 3 predators, 1 prey and "
                "2 landmarks; they differ only in the prey's policy, which scores candidate "
                "moves with JaxMARL's `score_function='sum'` here and `'min'` in 3a. On the "
                "same state and key the two policies chose different prey actions (largest "
                "difference per reset between 0.025 and 1.12 over 10 resets). "
                if name == "MPE_simple_facmac_v1"
                else "The prey's policy scores candidate moves with JaxMARL's "
                "`score_function='min'`; `MPE_simple_facmac_v1` has the same layout as 3a "
                "with `'sum'`. "
            )
            if facmac
            else ""
        )
        + (
            ""
            if solo
            else "Scripted slots look two ticks ahead on the learners' reward; sparse rewards "
            "mostly tie, so they act nearly at random there."
        ),
        guide="marl.md",
        history=(f"`{version_of(env_id)}`: wraps JaxMARL 0.2; the version follows JaxMARL's id.",),
    )


def content_for(env_id: str, solo: bool = False) -> Content:
    return ENVS[env_id] if env_id in ENVS else generic(env_id, solo)

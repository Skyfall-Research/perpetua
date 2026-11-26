"""Tutorial step 4: test, register and train.

``pp.testing.check_env`` runs the lifetime-contract checks on fresh instances
from a factory and names the first thing that breaks. Registering the factory
makes the world reachable by id, and every runner and learner then works on it
unchanged.
"""

import jax
import numpy as np
from tutorial3_drift_and_masks import make

import perpetua as pp
from perpetua.agents import PPO, PPOConfig, PPORunner

# The berry alternates between the right-hand corners while the left fence
# opens and closes, so the berry is always reachable.
drifts = {
    "berry": pp.drift.Cyclic([(4.0, 4.0), (0.0, 4.0)], period=2048),
    "fences": pp.drift.Cyclic([(1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)], period=512),
}

# 1. The contract: reproducibility, chunk continuity and the clock, privilege
#    isolation and the Oracle, shapes, state types, compilation, checkpoints.
pp.testing.check_env(lambda: make(drift=drifts))
print("check_env passed")

# 2. Registration: reach the world by id, as configuration files do.
pp.register("tutorial/BerryField-v0", make, overwrite=True)
env = pp.make("tutorial/BerryField-v0", size=5, drift=drifts)

# 3. Training: a continuing PPO learner, compared with random actions.
trainer = PPORunner(env, PPO(action_size=5, config=PPOConfig(rollout_steps=64)))
state = trainer.init(seed=0, window_size=1024)
for _ in range(160):
    state = trainer(state)
ticks = state.lifetime.state.t.to_int()
random = pp.runtime.Runner(env, pp.policies.random, ticks=ticks)(
    pp.runtime.init_lifetime(env, seed=0, window_size=1024)
)
# Lifetime means: performance is online reward, measured as it accrues.
learned = float(np.asarray(state.lifetime.metrics.mean)[0])
baseline = float(np.asarray(random.metrics.mean)[0])
print(f"after {ticks} ticks: PPO {learned:.4f} vs random {baseline:.4f} per tick")
assert learned > baseline
pp.unregister("tutorial/BerryField-v0")
jax.clear_caches()

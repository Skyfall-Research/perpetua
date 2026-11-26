"""Continuing MuJoCo control on MJX with drifting physics (gravity, mass).

MJX runs MuJoCo's dynamics in JAX, so these lifetimes compile, vmap and
checkpoint like any native environment. The models are small MJCF files
written for Perpetua (a swing-up pendulum and a swing-up cart-pole), not copies
of another suite's assets. There are no terminations or time limits: the cart
is held by joint limits and the pole can always be swung up again.

Two parameters drift through :class:`~perpetua.params.DriftEnv`: ``gravity``
(magnitude, m/s^2) and ``mass_scale`` (a multiplier on every body's mass and
inertia, applied also to the constants MuJoCo derives from them). Requires the
optional ``mjx`` extra (``mujoco-mjx``).
"""

# JAX 0.11 leaves device/tracer members partly untyped; other strict checks stay enabled.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportUnknownLambdaType=false

import importlib.metadata
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from perpetua.core import EnvInfo, Metadata

__all__ = [
    "TASKS",
    "MJXParams",
    "MJXState",
    "MJXControl",
]

PENDULUM = """
<mujoco model="perpetua-pendulum">
  <option timestep="0.01" integrator="RK4"/>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 1" size="0.04" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1" ctrllimited="true" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

CARTPOLE = """
<mujoco model="perpetua-cartpole">
  <option timestep="0.01" integrator="RK4"/>
  <worldbody>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2 2" damping="0.1"/>
      <geom type="box" size="0.2 0.1 0.05" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.8" size="0.03" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

TASKS = {"pendulum": PENDULUM, "cartpole": CARTPOLE}


class MJXParams(NamedTuple):
    """Gravity magnitude (1,) and body mass/inertia multiplier (1,), float32."""

    gravity: jax.Array
    mass_scale: jax.Array


class MJXState(NamedTuple):
    """Joint positions and velocities: the complete state of these models."""

    qpos: jax.Array
    qvel: jax.Array


def _wrap(angle: jax.Array) -> jax.Array:
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


class MJXControl:
    """A swing-up task on MJX under drifting gravity and mass.

    ``task="pendulum"``: observations float32 (1, 3) are cos and sin of the
    pole angle from upright and its angular velocity; the action is a torque in
    [-2, 2]; the reward is ``-(angle**2 + 0.1 * velocity**2 + 0.001 * torque**2)``
    (Gymnasium's Pendulum cost). ``task="cartpole"``: observations float32
    (1, 5) are cart position and velocity, cos and sin of the pole angle and its
    angular velocity; the action is a normalised force in [-1, 1]; the reward is
    ``cos(angle) - 0.01 * position**2 - 0.001 * force**2``. Each decision applies
    the action for ``frame_skip`` physics steps of 10 ms. Both start hanging
    with a seeded perturbation, and never terminate or respawn.
    """

    num_agents = 1
    default_drift_field = "gravity"
    drift_fields = ("gravity", "mass_scale")

    def __init__(self, *, task: str = "pendulum", frame_skip: int = 5) -> None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)}.")
        if type(frame_skip) is not int or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer.")
        import mujoco
        from mujoco import mjx

        self._mjx = mjx
        self.task = task
        self.frame_skip = frame_skip
        self.model = mujoco.MjModel.from_xml_string(TASKS[task])
        self._model = mjx.put_model(self.model)
        self._data = mjx.make_data(self._model)
        self._mass = jnp.asarray(self.model.body_mass, dtype=jnp.float32)
        self._inertia = jnp.asarray(self.model.body_inertia, dtype=jnp.float32)
        features = 3 if task == "pendulum" else 5
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1, features), dtype=np.float32)
        limit = 2.0 if task == "pendulum" else 1.0
        self.action_space = spaces.Box(-limit, limit, shape=(1, 1), dtype=np.float32)
        self.metadata: Metadata = {
            "env_id": f"mjx/{'Pendulum' if task == 'pendulum' else 'CartPole'}-v0",
            "oracle_exposure": (),
            "execution": "jax",
            "checkpointable": True,
            "configuration": {
                "backend": f"mujoco-mjx-{importlib.metadata.version('mujoco-mjx')}",
                "task": task,
                "frame_skip": frame_skip,
                "timestep": 0.01,
                "respawn_policy": "none-continuing",
                "drift_parameters": ["gravity", "mass_scale"],
            },
        }

    @property
    def default_params(self) -> MJXParams:
        """Earth gravity and the model's own masses."""
        return MJXParams(jnp.asarray([9.81], jnp.float32), jnp.asarray([1.0], jnp.float32))

    def derive(self, params: MJXParams) -> MJXParams:
        """Keep gravity non-negative and masses at least 5% of nominal."""
        return MJXParams(jnp.maximum(params.gravity, 0.0), jnp.maximum(params.mass_scale, 0.05))

    def _physics(self, params: MJXParams) -> Any:
        """The model under the drifted gravity and mass.

        Scaling every mass and inertia by ``s`` also scales the constants
        MuJoCo derives from them (``mj_setConst``): subtree masses, the
        joint-space inertia diagonal and the mean inertia by ``s``, and the
        inverse weights the constraint solver reads (joint limits) and the
        actuator acceleration by ``1 / s``. With ``s = 1`` every field keeps
        its value exactly.
        """
        model = self._model
        scale = params.mass_scale[0]
        return model.replace(
            opt=model.opt.replace(gravity=jnp.asarray([0.0, 0.0, -1.0]) * params.gravity[0]),
            body_mass=self._mass * scale,
            body_inertia=self._inertia * scale,
            body_subtreemass=model.body_subtreemass * scale,
            dof_M0=model.dof_M0 * scale,
            body_invweight0=model.body_invweight0 / scale,
            dof_invweight0=model.dof_invweight0 / scale,
            actuator_acc0=model.actuator_acc0 / scale,
            stat=model.stat.replace(meaninertia=model.stat.meaninertia * scale),
        )

    def init(self, key: jax.Array, params: MJXParams) -> tuple[MJXState, jax.Array, EnvInfo]:
        """Start hanging down with a small seeded angle and velocity perturbation."""
        angle_key, velocity_key = jax.random.split(key)
        qpos = (
            jnp.zeros(self.model.nq, jnp.float32)
            .at[-1]
            .set(jnp.pi + jax.random.uniform(angle_key, (), minval=-0.3, maxval=0.3))
        )
        qvel = jax.random.uniform(velocity_key, (self.model.nv,), minval=-0.3, maxval=0.3)
        state = MJXState(qpos, qvel.astype(jnp.float32))
        obs, info = self.observe(state, params)
        return state, obs, info

    def observe(self, state: MJXState, params: MJXParams) -> tuple[jax.Array, EnvInfo]:
        """Angles as cos/sin from upright, plus velocities (and cart position)."""
        del params
        angle = state.qpos[-1]
        pole = [jnp.cos(angle), jnp.sin(angle), state.qvel[-1]]
        values = pole if self.task == "pendulum" else [state.qpos[0], state.qvel[0], *pole]
        info: EnvInfo = {
            "action_mask": jnp.ones((1, 1), dtype=jnp.bool_),
            "active_mask": jnp.asarray([True]),
            "respawn": jnp.asarray([False]),
        }
        return jnp.stack(values).astype(jnp.float32)[None], info

    def step(
        self, state: MJXState, action: Any, key: jax.Array, params: MJXParams
    ) -> tuple[MJXState, jax.Array]:
        """Apply the clipped action for ``frame_skip`` physics steps."""
        del key
        action = jnp.asarray(action)
        if action.shape != (1, 1) or action.dtype.kind != "f":
            raise ValueError("MJX control actions must be float arrays of shape (1, 1).")
        control = jnp.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        model = self._physics(params)
        # MJX computes in the dtype it built its data with (float64 under x64).
        dtype = self._data.qpos.dtype
        data = self._data.replace(
            qpos=state.qpos.astype(dtype), qvel=state.qvel.astype(dtype), ctrl=control.astype(dtype)
        )
        data = jax.lax.fori_loop(
            0, self.frame_skip, lambda _, current: self._mjx.step(model, current), data
        )
        qpos, qvel = data.qpos.astype(jnp.float32), data.qvel.astype(jnp.float32)
        angle = _wrap(qpos[-1])
        if self.task == "pendulum":
            reward = -(angle**2 + 0.1 * qvel[-1] ** 2 + 0.001 * control[0] ** 2)
        else:
            reward = jnp.cos(angle) - 0.01 * qpos[0] ** 2 - 0.001 * control[0] ** 2
        return MJXState(qpos, qvel), jnp.asarray(reward, jnp.float32)[None]

    def privileged(self, state: MJXState) -> dict[str, jax.Array]:
        """No world-side diagnostics; the wrapper supplies the drift ones."""
        del state
        return {}

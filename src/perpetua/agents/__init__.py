"""Optional learning agents; install Perpetua's training extra before importing."""

from perpetua.core import requires_extra

with requires_extra("training", "perpetua.agents"):
    from perpetua.agents.compute_ppo import ComputePPORunner
    from perpetua.agents.dqn import (
        DQN,
        IQL,
        DQNConfig,
        DQNHyperparameters,
        DQNRunner,
        HostDQNRunner,
    )
    from perpetua.agents.ppo import (
        IPPO,
        PPO,
        HostPPORunner,
        PPOConfig,
        PPOHyperparameters,
        PPORunner,
    )
    from perpetua.agents.sac import (
        SAC,
        HostSACRunner,
        SACConfig,
        SACHyperparameters,
        SACRunner,
    )

__all__ = [
    "ComputePPORunner",
    "DQN",
    "DQNConfig",
    "DQNHyperparameters",
    "DQNRunner",
    "HostDQNRunner",
    "HostPPORunner",
    "HostSACRunner",
    "IQL",
    "IPPO",
    "PPO",
    "PPOConfig",
    "PPOHyperparameters",
    "PPORunner",
    "SAC",
    "SACConfig",
    "SACHyperparameters",
    "SACRunner",
]

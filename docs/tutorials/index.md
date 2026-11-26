# Tutorials

Each tutorial teaches one thing you can reuse, shows its real output and runs
in seconds to a minute on a laptop CPU. Start with the
[quickstart](quickstart.md); the rest can be read in any order, but this order
builds up gradually.

- [Drifting a gymnax environment](gymnax-environment.md): change an existing
  environment's physics during the lifetime.
- [PPO in an open-ended world](bean-world-ppo.md): train in a world that never
  ends, then change what it rewards and watch the learner recover.
- [A custom drift process](custom-drift.md): write your own rule for how a
  parameter changes.
- [Several tasks in one lifetime](task-switching.md): switch between whole
  environments on a schedule.
- [Comparing runs](evaluation.md): seeds, per-phase scores, IQM and
  confidence intervals.
- [Average reward with PPO](reward-centering.md): what reward centering
  changes and when it matters.
- [SAC on Continual World](continual-world-sac.md): continuous control on a
  task-sequence benchmark (needs Meta-World).
- [Custom environment tutorial](custom-environment/index.md): four steps from a
  skeleton to a tested, registered world a learner trains in.

The test suite runs every code block on these pages.

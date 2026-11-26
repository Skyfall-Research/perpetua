# Custom environment tutorial

This tutorial builds `BerryField`, a small continual world, from an empty
skeleton to a tested, registered environment that a PPO learner trains in. It
follows the same path as PettingZoo's custom-environment tutorial, with the
differences a lifetime brings: no `reset`, respawns as events, and change over
time coming from drift on parameters.

1. [Project structure](1-project-structure.md): the files and the methods.
2. [Environment logic](2-environment-logic.md): a working world.
3. [Drift and evolving actions](3-drift-and-masks.md): fences that open and
   close, masking moves.
4. [Testing your environment](4-testing.md): the contract checker,
   registration and training.

Each step is a complete file under `examples/custom_environment/`, run by the
test suite. For the full method reference, see
[environment creation](../../custom-environments.md); for everything on one
page, see [a custom environment in one page](../custom-environment.md).

```{toctree}
:hidden:

1-project-structure
2-environment-logic
3-drift-and-masks
4-testing
```

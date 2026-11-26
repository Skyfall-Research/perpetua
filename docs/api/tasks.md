# pp.tasks

Several environments in one lifetime, selected by a drift over the task index.
A task that brings drift of its own is refused; a task's parameters start from
their defaults and may drift through the switch's mapping
(`tasks[i].<field>`). Every change of the active task is a declared switch.
Host lifts can be switched with `on_switch="resume"` only.
See [task switching](../task-switching.md).

```{eval-rst}
.. automodule:: perpetua.tasks
   :members:
```

# How to cite

If you use Perpetua in your research, please cite the software:

```bibtex
@misc{perpetua,
  title        = {Perpetua: Never-Ending, Changing Worlds for Continual Reinforcement Learning},
  author       = {{Perpetua contributors}},
  year         = {2026},
  howpublished = {\url{https://github.com/Skyfall-Research/perpetua}},
  note         = {Version 0.0.1a0}
}
```

Replace the version with the one you used
(`python -c "import importlib.metadata as m; print(m.version('perpetua-rl'))"`),
and give the source revision if you ran from a checkout
(`pp.reporting.provenance()["revision"]`).

## Cite the benchmarks you use

Perpetua wraps other people's benchmarks and does not replace their papers.
When you report results on one, cite it as well. Each environment page links
to its upstream project in the "Upstream" column of the
[environment index](../environments/index.md); for the continual benchmarks:

| Benchmark | Cite |
|---|---|
| Continual World | Wołczyk et al. (2021), and [Meta-World](https://metaworld.farama.org/) |
| COOM | the [COOM project](https://github.com/TTomilin/COOM) |
| MEAL | the [MEAL project](https://github.com/TTomilin/MEAL) |
| AgarCL | the [AgarCL project](https://github.com/AgarCL/AgarCL) |
| MORPHEUS | [the MORPHEUS paper](https://openreview.net/forum?id=31P1VAfLkJ) |
| MPE tasks | [MPE2](https://mpe2.farama.org/) and [JaxMARL](https://github.com/bold-lab-ai/JaxMARL) |
| gymnax environments | [gymnax](https://github.com/RobertTLange/gymnax) and the original suite (MinAtar, bsuite, Gymnasium) |

The bundled PPO adapts PureJaxRL; see [training](../training.md#provenance-and-limits).

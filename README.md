# Inverse RL

Implementations for imitation learning / IRL algorithms in RLLAB

Contains:
- GAIL (https://arxiv.org/abs/1606.03476/pdf)
- Guided Cost Learning (https://arxiv.org/pdf/1611.03852.pdf)
- Tabular MaxEnt IRL (https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)

Setup
---
This library requires:
- rllab (https://github.com/openai/rllab)
- Tensorflow

Examples
---

Running the Pendulum-v0 gym environment:

1) Collect expert data
```
python scripts/pendulum_data_collect.py
```

You should get an "AverageReturn" of around -100 to -150

2) Run imitation learning
```
python scripts/pendulum_gcl.py
```

The "OriginalTaskAverageReturn" should reach around -100 to -150

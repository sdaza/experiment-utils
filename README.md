# Experimental utils

Generic functions for experiment analysis and design

- [Experimental utils](#experimental-utils)
- [Installation](#installation)
- [How to use it](#how-to-use-it)
	- [Power Analysis](#power-analysis)

# Installation 

```
pip install git+https://github.com/sdaza/experiment-utils.git
```

# How to use it

## Power Analysis


```python
from experiment_utils import PowerSim
p = PowerSim(metric='proportion', relative_effect=False,
	variants=1, nsim=1000, alpha=0.05, alternative='two-tailed')

p.get_power(baseline=[0.33], effect=[0.03], sample_size=[3000])
```


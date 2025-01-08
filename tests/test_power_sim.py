import pytest
from experiment_utils.power_sim import PowerSim
from experiment_utils.spark_instance import *


def test_power_estimation():
    """Test power estimation"""
    p = PowerSim(metric='proportion', relative_effect=False, variants=1,
                 nsim=1000, alpha=0.05, alternative='two-tailed')
    try:
        p.get_power(baseline=[0.33], effect=[0.03], sample_size=[3000])
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_plot_power():
    """Test plot power"""
    p = PowerSim(metric='proportion', relative_effect=False,
                 variants=2, alternative='two-tailed',
                 nsim=100, correction='holm')
    try:
        p.grid_sim_power(baseline_rates=[[0.33]],
                         effects=[[0.01, 0.03], [0.03, 0.05], [0.03, 0.07]],
                         sample_sizes=[[1000], [5000], [9000]],
                         threads=16,
                         plot=False)
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")

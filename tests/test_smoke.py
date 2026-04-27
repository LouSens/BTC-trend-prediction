"""Sanity check that the package imports."""
import mcmc_cuda


def test_version_string():
    assert isinstance(mcmc_cuda.__version__, str)
    assert mcmc_cuda.__version__.count(".") == 2

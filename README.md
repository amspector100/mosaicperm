A python implementation of the mosaic permutation testing framework.

# Installation

To install ``mosaicperm``, just use pip:

``pip install mosaicperm``

# Documentation

Documentation and tutorials are available at https://mosaicperm.readthedocs.io/.

# Quickstart

Below, we give a simple example showing how to use ``mosaicperm`` to test whether a set of factor exposures explain the correlations among a matrix of outcomes variables.

```
	import numpy as np
	import mosaicperm as mp

	# synthetic outcomes and exposures
	n_obs, n_subjects, n_factors = 100, 200, 20
	outcomes = np.random.randn(n_obs, n_subjects)
	exposures = np.random.randn(n_obs, n_subjects, n_factors)
	# example of missing data
	outcomes[0:10][:, 0:5] = np.nan
	exposures[0:10][:, 0:5] = np.nan

	# fit mosaic permutation test
	mpt = mp.factor.MosaicFactorTest(
		outcomes=outcomes,
		exposures=exposures,
		test_stat=mp.statistics.mean_maxcorr_stat,
	)
	print(mpt.fit().summary())

	# produce a time series plot of this analysis
	mpt.fit_tseries(
		nrand=100, n_timepoints=20,
	).plot_tseries()
```

See the [documentation](https://mosaicperm.readthedocs.io/) for more details.
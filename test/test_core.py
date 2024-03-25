import time
import numpy as np
import unittest
import pytest
import os
import sys
from scipy import stats
try:
	from . import context
	from .context import mosaicperm as mp
# For profiling
except ImportError:
	import context
	from context import mosaicperm as mp

def adaptive_tstat(x):
	"""
	"""
	mu0 = x.mean(axis=0)
	sd0 = x.std(axis=0)
	return np.around(np.max((x[0] - mu0) / sd0), 5) # to prevent numerical errors

class TestAdaptivePval(context.MosaicTest):
	"""
	tests core.adaptive_pval functions
	"""

	def test_cond_on_data(self):
		np.random.seed(123)
		# Create data
		reps, nrand, d = 10000, 5, 50
		adapt_stats = np.random.randn(nrand, d)
		# Check that the p-value is correct conditional on the data
		pval = mp.core.compute_adaptive_pval(
			adapt_stats[0], adapt_stats[1:]
		)[0]
		stat = adaptive_tstat(adapt_stats)
		null_stats = np.zeros(reps)
		inds = np.arange(nrand)
		for r in range(reps):
			np.random.shuffle(inds)
			null_stats[r] = adaptive_tstat(adapt_stats[inds])
		pval_expected = (1 + np.sum(stat <= null_stats)) / (1 + reps)
		np.testing.assert_array_almost_equal(
			pval,
			pval_expected,
			decimal=2,
			err_msg=f"compute_adaptive_pval gives unexpected result conditional on the data"
		)

	def test_empirical_validity(self):
		"""
		Tests
		"""
		# AR1 correlation matrix 
		rho, reps, nrand, d = 0.9, 1000, 2, 20
		inds = np.arange(d)
		dists = np.abs(inds.reshape(-1, 1) - inds.reshape(1, -d))
		C1 = rho**(dists) # correlation matrix
		L1 = np.linalg.cholesky(C1)
		# exchangeable correlation matrix
		C2 = rho * np.ones((nrand, nrand)) + (1 - rho) * np.eye(nrand) 
		L2 = np.linalg.cholesky(C2)
		# Create data
		pvals = np.zeros(reps)
		for r in range(reps):
			adapt_stats = L2 @ (np.random.randn(nrand, d) @ L1.T)
			pvals[r] = mp.core.compute_adaptive_pval(
				adapt_stats[0], adapt_stats[1:]
			)[0]
		self.check_binary_pval_symmetric(pvals, alpha=0.001)
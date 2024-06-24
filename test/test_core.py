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

class TestCombinations(context.MosaicTest):
	"""
	Tests functions which combine mosaic tests.
	"""

	def test_combination_functions(self):
		# Create data
		np.random.seed(123)
		n_obs, n_subjects, n_factors = 200, 50, 2
		exposures = np.random.randn(n_obs, n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects) 
		outcomes += np.arange(n_obs).reshape(-1, 1) * np.random.randn(n_obs, 1) / n_obs
		# Fit many mosaic models
		n_seeds, n_timepoints, nrand, window = 3, 5, 20, 50
		for test_stat, adaptive in zip(
			[mp.statistics.mean_maxcorr_stat, mp.statistics.quantile_maxcorr_stat],
			[False, True]
		):
			mpts = []
			for seed in range(n_seeds):
				mpt = mp.factor.MosaicFactorTest(
					outcomes=outcomes, exposures=exposures, test_stat=test_stat,
					seed=seed,
				).fit_tseries(nrand=nrand, n_timepoints=n_timepoints, verbose=False, window=window)
				mpt.fit(nrand=nrand, verbose=False)
				mpts.append(mpt)
			# Check that the combination functions work
			for how in ['mean', 'median', 'min']:
				# Non t-series variant
				output = mp.core.combine_mosaic_tests(
					mosaic_objects=mpts, how_combine_pvals=how,
				)
				df = mp.core.combine_mosaic_tests_tseries(
					mosaic_objects=mpts, how_combine_pvals=how, plot=True, show_plot=False,
				)
				# Check that everything looks correct
				zstat = output['apprx_zstat'].item()
				expected = np.mean([mpt.apprx_zstat for mpt in mpts])
				self.assertTrue(
					zstat == expected,
					f"zstat {zstat} is not the mean of the apprx zstats {expected}"
				)
				zstats = df['apprx_zstat'].values
				expected = np.stack([mpt.apprx_zstat_tseries for mpt in mpts], axis=0).mean(axis=0)
				np.testing.assert_array_almost_equal(
					zstats, expected, decimal=6, err_msg=f"Mean of zstats seems incorrect"
				)
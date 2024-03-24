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

def fast_maxcorr_stat(residuals):
	"""
	Computes max correlation. No zero-SD checks for speed.
	Adds a tiny bit of noise to break ties.
	"""
	return np.abs(
		np.corrcoef(residuals.T) - np.eye(residuals.shape[1])
	).max() + np.random.uniform(0, 0.001)

class TestMosaicFactorTest(context.MosaicTest):
	"""
	tests tiling functions
	"""

	def test_residual_orthogonality(self):
		n_obs, n_subjects, n_factors = 150, 200, 20
		for nstart in [0, 5, 15, 50]:
			# Create exposure data
			if nstart > 0:
				exposures = context._create_exposures(
					n_obs=n_obs, n_subjects=n_subjects, n_factors=n_factors, nstart=nstart,
				)
			else:
				exposures = np.random.randn(n_subjects, n_factors)
			outcomes = np.random.randn(n_obs, n_subjects)
			# Construct residuals
			mpt = mp.factor.MosaicFactorTest(outcomes=outcomes, exposures=exposures, test_stat=None)
			mpt.compute_mosaic_residuals()
			mpt.permute_residuals()
			# Check orthogonality
			for resids, name in zip([mpt.residuals, mpt._rtilde], ['Residuals', 'Permuted residuals']):
				for (batch, group) in mpt.tiles:
					for j in batch:
						if nstart > 0:
							tilej_exposures = exposures[j, group]
						else:
							tilej_exposures = exposures[group]
						np.testing.assert_array_almost_equal(
							tilej_exposures.T @ resids[j, group],
							np.zeros(n_factors),
							decimal=5,
							err_msg=f"{name} for j={j}, group={group} are not orthogonal to exposures with nstart={nstart}."
						)

	def test_reduction_to_ols(self):
		"""
		Tests that with one tile and constant exposures, the residuals are OLS residuals.
		"""
		n_obs, n_subjects, n_factors = 150, 200, 20
		exposures = np.random.randn(n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		tiles = [(np.arange(n_obs), np.arange(n_subjects))]
		## compute mosaic residuals with one large tile
		mpt = mp.factor.MosaicFactorTest(
			outcomes=outcomes, exposures=exposures, test_stat=None, tiles=tiles
		)
		mpt.compute_mosaic_residuals()
		## Compute OLS residuals
		# Use inversion instead of QR just to add variety
		H = exposures @ np.linalg.inv(exposures.T @ exposures) @ exposures.T
		H = np.eye(n_subjects) - H
		ols_resid = outcomes @ H
		np.testing.assert_array_almost_equal(
			mpt.residuals, ols_resid, decimal=5, err_msg=f"Mosaic residuals do not reduce to OLS resids with one tile"
		)

	def test_permute_residuals(self):
		# data
		n_obs, n_subjects, n_factors = 150, 200, 20
		exposures = np.random.randn(n_subjects, n_factors)
		outcomes = np.random.randn(n_obs, n_subjects)
		# Construct and permute residuals
		mpt = mp.factor.MosaicFactorTest(outcomes=outcomes, exposures=exposures, test_stat=None)
		mpt.compute_mosaic_residuals()
		mpt.permute_residuals()
		# check
		np.testing.assert_array_almost_equal(
			np.sort(np.unique(mpt.residuals)), 
			np.sort(np.unique(mpt._rtilde)),
			decimal=5,
			err_msg=f"mpt residuals and _rtilde do not have the same elements in different orders"
		)
		self.assertTrue(
			np.any(mpt.residuals != mpt._rtilde),
			"After permuting, _rtilde=residuals everywhere."
		)

	def test_binary_pval(self):
		"""
		For maximum efficiency, uses small scale data to test whether
		the p-value with one randomization is valid.
		"""
		# Very small scale data
		np.random.seed(123)
		reps = 2000
		n_obs, n_subjects, n_factors = 10, 20, 2
		pvals = np.zeros(reps)
		# Factors and exposures
		exposures = stats.laplace.rvs(size=(n_subjects, n_factors))
		factors = stats.laplace.rvs(size=(n_obs, n_factors))
		mu = factors @ exposures.T

		# loop through
		for r in range(reps):
			outcomes = stats.laplace.rvs(size=(n_obs, n_subjects)) + mu 
			# construct
			mpt = mp.factor.MosaicFactorTest(
				outcomes=outcomes,
				exposures=exposures,
				test_stat=fast_maxcorr_stat,
			) 
			mpt.fit(nrand=1, verbose=False)
			pvals[r] = mpt.pval

		self.check_binary_pval_symmetric(pvals, alpha=0.001)
import time
import numpy as np
import pandas as pd
import math
import unittest
import pytest
import os
import sys
from scipy import stats
import scipy.sparse
try:
	from . import context
	from .context import mosaicperm as mp
# For profiling
except ImportError:
	import context
	from context import mosaicperm as mp

def mmc_stat(residuals, times, subjects, **kwargs):
	"""
	This uses a particular reshaping trick for efficiency.
	It should not be used in general (use the commented out code instead).
	"""
	residuals = residuals.reshape(np.max(times) +1, np.max(subjects)+1, order='C')
	corrs = np.corrcoef(residuals.T)
	# corrs = pd.DataFrame(
	# 	np.stack([residuals, times, subjects], axis=1),
	# 	columns=['residuals', 'times', 'subjects']
	# ).pivot(
	# 	index='times',
	# 	columns='subjects',
	# 	values='residuals'
	# ).corr().values
	return np.abs(corrs - np.eye(corrs.shape[0])).max(axis=0).mean()

def quadratic_cov_stat(residuals, times, subjects, **kwargs):
	"""
	This uses a particular reshaping trick for efficiency.
	It should not be used in general (use the commented out code instead).
	"""
	# rdf = pd.DataFrame(
	# 	np.stack([residuals, times, subjects], axis=1),
	# 	columns=['residuals', 'times', 'subjects']
	# ).pivot(
	# 	index='times',
	# 	columns='subjects',
	# 	values='residuals'
	# )
	# covs = rdf.values.T @ rdf.values
	residuals = residuals.reshape(np.max(times) +1, np.max(subjects)+1, order='C')
	covs = residuals.T @ residuals
	return np.mean(covs - np.diag(np.diag(covs)))



class TestTileTransformations(context.MosaicTest):

	def test_transformations_automatic(self):
		np.random.seed(123)
		# fake data
		n = 300
		subjects = np.random.randint(0, 20, n)
		times = np.random.choice(2*n, n, replace=False)
		covariates = np.random.randn(n, 1)
		args = dict(subjects=subjects, times=times, covariates=covariates)
		# transformations
		invariances = np.array(['local_exch', 'local_exch_cts', 'time_reverse', 'wild_bs'])
		transformations = []
		for invariance in invariances:
			transformations.append(mp.panel.construct_tile_transformation(
				**args, invariance=invariance
			))
		# Test
		residuals = np.random.randn(n)
		for invariance, transform in zip(invariances, transformations):
			trans_resid = transform(residuals)
			trans_times = transform(times)
			np.testing.assert_array_almost_equal(
				residuals,
				transform(trans_resid),
				decimal=8,
				err_msg=f"transform={invariance} does not satisfy transform^2=identity"
			)
			# Test that we preserve within-subject residuals
			if invariance != 'wild_bs':
				for subject in np.unique(subjects):
					# look at time series for specific subject
					inds = np.where(subjects == subject)[0]
					subject_resid = residuals[inds]
					subject_trans_resid = trans_resid[inds]
					np.testing.assert_array_almost_equal(
						np.sort(subject_resid),
						np.sort(subject_trans_resid),
						decimal=8,
						err_msg=f"{invariance} does not preserve unique values of residuals within subjects"
					)
					# sorted times within subject
					stimes_argsort = np.argsort(times[inds])
					# test correctness for local exchangeability
					if invariance == 'local_exch_cts':
						n_pairs_to_check = int(len(stimes_argsort) // 2)
						for k in range(n_pairs_to_check):
							# check that the k-1th and kth smallest times are swapped
							within_subject_inds = stimes_argsort[2*k:(2*k+2)]
							np.testing.assert_array_almost_equal(
								times[inds][within_subject_inds],
								np.flip(trans_times[inds][within_subject_inds]),
								decimal=8,
								err_msg="local_exch_cts transformation does not swap the smallest and second smallest times"
							)
					# Test correctness for time-reversibility
					if invariance == 'time_reverse':
						np.testing.assert_array_almost_equal(
							times[inds][stimes_argsort],
							np.flip(trans_times[inds][stimes_argsort]),
							decimal=8,
							err_msg="time_reverse transformation does not reverse time properly"
						)
			# Test correctness for local exch
			if invariance == 'local_exch':
				self.assertTrue(
					np.abs(times - trans_times).max() <= 1,
					"local_exch times and trasnformed times differ by more than 1"
				)
				self.assertTrue(
					np.all(
						(trans_times - times != 1) | (times % 2 == 0)
					),
					"local_exch swaps odd t with t+1 (should only do even t)"
				)
				self.assertTrue(
					np.all(
						(trans_times - times != -1) | (times % 2 == 1)
					),
					"local_exch swaps even t with t-1 (should only do odd t)"
				)
			# Test correctness for wild_bs
			if invariance == 'wild_bs':
				np.testing.assert_array_almost_equal(
					-1*residuals,
					trans_resid,
					decimal=8,
					err_msg=f"wild_bs does not flip sign of residuals"
				)

class TestMosaicPanelTest(context.MosaicTest):
	"""
	tests mosaic factor test class
	"""
	def test_residual_orthogonality(self):
		n_obs, n_subjects, n_cov = 15, 200, 20
		# Create outcomes and covariates
		outcomes = np.random.randn(n_obs * n_subjects)
		subjects = np.arange(n_obs * n_subjects) // n_obs
		times = np.arange(n_obs * n_subjects) - subjects
		covariates = np.random.randn(n_obs * n_subjects, n_cov)
		# loop through invariances
		for invariance in ['local_exch_cts', 'time_reverse', 'wild_bs']:
			# Construct residuals
			mpt = mp.panel.MosaicPanelTest(
				outcomes=outcomes,
				subjects=subjects,
				times=times,
				cts_covariates=covariates,
				test_stat=None,
				invariance=invariance,
			)
			mpt.compute_mosaic_residuals()
			mpt.permute_residuals()
			# Check orthogonality
			for resids, name in zip([mpt.residuals, mpt._rtilde], ['Residuals', 'Permuted residuals']):
				for tile_id, tile_inds in enumerate(mpt.tile_inds):
					tile_cov = covariates[tile_inds]
					tile_resids = resids[tile_inds]
					# Test orthogonality
					np.testing.assert_array_almost_equal(
						tile_cov.T @ tile_resids,
						np.zeros(n_cov),
						decimal=5,
						err_msg=f"{name} for tile_inds={tile_inds} are not orthogonal to covariates."
					)
					# test orthogonality with transformed covariates
					np.testing.assert_array_almost_equal(
						mpt.tile_transforms[tile_id](tile_cov).T @ tile_resids,
						np.zeros(n_cov),
						decimal=5,
						err_msg=f"{name} for tile_inds={tile_inds} are not orthogonal to covariates."
					)

	def test_reduction_to_ols(self):
		"""
		Tests that with one tile, the residuals are OLS residuals 
		if the design does not change except every two observations.
		"""
		# Create data
		n_obs, n_subjects, n_cov = 10, 50, 5
		outcomes = np.random.randn(n_obs, n_subjects)
		covariates = np.random.randn(n_obs, n_subjects, n_cov)
		# Ensure covariates is the same as its augmented variant
		swapinds = mp.panel._construct_swapinds(n_obs)
		covariates = np.around(covariates + covariates[swapinds], 5)
		# Convert to be flattened
		subjects = np.stack([np.arange(n_subjects) for _ in range(n_obs)], axis=0).flatten(order='C')
		times = np.stack([np.arange(n_subjects) for _ in range(n_obs)], axis=0).flatten(order='C')
		outcomes = outcomes.flatten(order='C')
		covariates = covariates.reshape(-1, n_cov, order='C')

		## compute mosaic residuals with one large tile
		mpt = mp.panel.MosaicPanelTest(
			outcomes=outcomes, subjects=subjects, times=times, cts_covariates=covariates, test_stat=None, ntiles=1,
			invariance='local_exch_cts'
		)
		mpt.compute_mosaic_residuals()
		## Compute OLS residuals
		ols_resid = mp.panel.ols_residuals_panel(outcomes, covariates=covariates)
		# debugging
		inds = mpt.tile_inds[0]
		hm = mp.panel.ols_residuals_panel(
			outcomes=mpt.outcomes[inds], # this is flat, still fine with this function call
			covariates=mpt.covariates[inds],
		)
		# end debugging
		np.testing.assert_array_almost_equal(
			mpt.residuals, ols_resid, decimal=5, err_msg=f"Mosaic residuals do not reduce to OLS resids with one tile"
		)

	def test_permute_residuals(self):
		# data
		np.random.seed(123)
		data = mp.gen_data.gen_panel_data(
			n_obs=150,
			n_subjects=150,
			n_cov=20,
			flat=True,
		)
		for ngroups in [1, 10, 20]:
			# Construct and permute residuals
			mpt = mp.panel.MosaicPanelTest(
				outcomes=data['outcomes'],
				times=data['times'],
				subjects=data['subjects'],
				cts_covariates=data['covariates'],
				test_stat=None,
				ntiles=ngroups,
				invariance='local_exch_cts',
			)
			mpt.compute_mosaic_residuals()
			mpt.permute_residuals()
			# check equality within tiles
			for tilenum, tile_inds in enumerate(mpt.tile_inds):
				# Same elements in different order
				np.testing.assert_array_almost_equal(
					np.sort(np.unique(mpt.residuals[tile_inds])), 
					np.sort(np.unique(mpt._rtilde[tile_inds])),
					decimal=5,
					err_msg=f"for tilenum={tilenum}, mpt residuals and _rtilde do not have the same elements in different orders"
				)
				# The tiles have the same rows but in different orders
				tile = mpt.residuals[tile_inds]
				tile_perm = mpt._rtilde[tile_inds]
				intersection = np.unique(np.concatenate([tile, tile_perm], axis=0), axis=0)
				self.assertTrue(
					intersection.shape == tile.shape,
					f"For ngroups={ngroups}, tile={tilenum}, the permuted tile does not have the same rows as the original tile"
				)

		# Make sure permuting does something
		self.assertTrue(
			np.any(mpt.residuals != mpt._rtilde),
			"After permuting, _rtilde=residuals everywhere."
		)
				
	def test_exact_pval(self):
		"""
		For maximum efficiency, uses small scale data to test whether
		the p-value is valid.
		"""
		# Very small scale data
		np.random.seed(123)
		reps = 500
		nrand = 9
		n_obs, n_subjects, n_cov = 20, 15, 2
		# data
		covariates = stats.laplace.rvs(size=(n_obs * n_subjects, n_cov))
		beta = stats.laplace.rvs(size=n_cov)
		mu = covariates @ beta
		# missing data pattern
		missing_flags = np.random.randn(n_obs * n_subjects)
		missing_flags = missing_flags < -1.25 

		# Including simulation setting with data missing at random
		for missing_data in [False]:
			pvals = np.zeros(reps)
			test_stats = np.zeros(reps)
			for r in range(reps):
				np.random.seed(r)
				data = mp.gen_data.gen_panel_data(n_obs=n_obs, n_subjects=n_subjects)
				outcomes = data['outcomes'] + mu
				if missing_data:
					outcomes[missing_flags] = np.nan

				# construct
				mpt = mp.panel.MosaicPanelTest(
					outcomes=outcomes,
					times=data['times'],
					subjects=data['subjects'],
					cts_covariates=covariates,
					test_stat=mmc_stat,
					tstat_kwargs=dict(times=data['times'], subjects=data['subjects'])
				) 
				mpt.fit(nrand=nrand, verbose=False)
				test_stats[r] = mpt.statistic
				pvals[r] = mpt.pval

			buckets = np.around(pvals * (nrand + 1)).astype(int)
			counts = np.bincount(buckets)
			print(pvals, counts)
			self.assertTrue(
				counts[0] == 0,
				f"with {nrand} randomizations, mosaic panel test produces a p-value < 1/(nrand + 1)"
			)
			counts = counts[1:]
			self.assertTrue(
				len(counts) == nrand + 1,
				f"with {nrand} randomizations, pvals only has {len(counts)} unique values (unique pvals={np.unique(pvals)})"
			)
			print("Pvals\n", pvals)
			print("tstats\n", test_stats)
			print(f"Missing={missing_data}")
			self.pearson_chi_square_test(counts, test_name='MosaicPanelTest p-values', alpha=0.001)

	def test_quadratic_test(self):
		nrand = 20000
		data = mp.gen_data.gen_panel_data(n_obs=10, n_subjects=20, n_cov=2)
		# Original variant
		mpt = mp.panel.MosaicPanelTest(
			outcomes=data['outcomes'],
			times=data['times'],
			subjects=data['subjects'],
			clusters=data['subjects'],
			cts_covariates=data['covariates'],
			test_stat=quadratic_cov_stat,
			tstat_kwargs=dict(times=data['times'], subjects=data['subjects']),
			ntiles=20,
		).fit(nrand=nrand)
		qmpt = mp.panel.QuadraticMosaicPanelTest(
			outcomes=data['outcomes'],
			times=data['times'],
			subjects=data['subjects'],
			clusters=data['subjects'],
			cts_covariates=data['covariates'],
			method='cov',
			ntiles=20,
		).fit(nrand=nrand)
		for mpt_val, qmpt_val, name, decimal in zip(
			[mpt.statistic, mpt.pval], [qmpt.statistic, qmpt.pval], ['statistic', 'pval'], [8, 2],
		):
			np.testing.assert_array_almost_equal(
				mpt_val,
				qmpt_val,
				decimal=decimal,
				err_msg=f"QuadraticMosaicPanelTest {name} does not match a manual implementation."
			)

class TestMosaicPanelInference(context.MosaicTest):

	def test_residual_orthogonality(self):
		n_cov = 5
		data = mp.gen_data.gen_panel_data(n_obs=50, n_subjects=20, n_cov=n_cov)
		for storage_type, covariates in zip(
			['dense', 'sparse'],
			[data['covariates'], scipy.sparse.csr_matrix(data['covariates'])],
		):
			# Construct original residuals
			mpi = mp.panel.MosaicPanelInference(
				outcomes=data['outcomes'],
				times=data['times'],
				subjects=data['subjects'],
				clusters=data['subjects'],
				cts_covariates=covariates,
			)
			mpi.compute_mosaic_residuals()
			for (tile_id, tile_inds) in enumerate(mpi.tile_inds):
				tile_cov = data['covariates'][tile_inds]
				tile_resids = mpi.residuals[tile_inds]
				# Test orthogonality
				np.testing.assert_array_almost_equal(
					tile_cov.T @ tile_resids,
					np.zeros(n_cov),
					decimal=5 if storage_type == 'dense' else 4,
					err_msg=f"MPI residuals for tile={tile_id} are not orthogonal to covariates with storage_type={storage_type}."
				)
			# Test residual orthogonality
			for feature in range(n_cov):
				# Compare LFO residuals to a naively computed variant
				lfo_resids = mpi._downdate_mosaic_residuals(feature=feature)
				neg_feature = [i for i in range(n_cov) if i != feature]
				mpi2 = mp.panel.MosaicPanelInference(
					outcomes=data['outcomes'],
					times=data['times'],
					subjects=data['subjects'],
					clusters=data['subjects'],
					cts_covariates=data['covariates'][..., neg_feature],
					tile_ids=mpi.tile_ids,
				)
				expected = mpi2.compute_mosaic_residuals()
				np.testing.assert_array_almost_equal(
					lfo_resids,
					expected,
					decimal=8 if storage_type == 'dense' else 4,
					err_msg=f"LFO resids are incorrect with feature={feature} and storage_type={storage_type}."
				)

	def test_equivariance(self):
		"""
		Tests that modifying y = y + b * X[:, 0]
		increases the estimate by b and does not change
		the standard error.
		"""
		# Generate data
		np.random.seed(123)
		nrand, alpha = 10000, 0.05
		features = [0,1]
		data = mp.gen_data.gen_panel_data(flat=True, n_obs=100, n_subjects=50, n_cov=10)
		# Fit original
		np.random.seed(123)
		mpt = mp.panel.MosaicPanelInference(
			outcomes=data['outcomes'],
			times=data['times'],
			subjects=data['subjects'],
			cts_covariates=data['covariates'],
			ntiles=10
		).fit(nrand=nrand, alpha=alpha, features=features)
		# Fit shifted
		b = np.random.randn()
		np.random.seed(123)
		mpt2 = mp.panel.MosaicPanelInference(
			outcomes=data['outcomes'] + b * data['covariates'][:, 0],
			times=data['times'],
			subjects=data['subjects'],
			cts_covariates=data['covariates'],
			tile_ids=mpt.tile_ids
		).fit(nrand=nrand, alpha=alpha, features=features)
		# Test estimates and SES
		for objtype in ['Estimate', 'Lower', 'Upper', 'SE']:
			for feature in features:
				obj1 = mpt.summary[objtype][feature]
				obj2 = mpt2.summary[objtype][feature]
				expected_diff = b if (feature == 0 and objtype != 'SE') else 0
				np.testing.assert_array_almost_equal(
					obj1,
					obj2 - expected_diff,
					decimal=5,
					err_msg=f"After replacing y with y + b * X[:, 0], {objtype} changes from {obj1} to {obj2} with b={b}"
				)


	def test_slope_intercept_representation(self):
		"""
		Tests that the test statistic for testing H_0 : beta[j] = b
		can be written as stat - b * slope for stat, slope.
		"""
		np.random.seed(1234)
		# Generate data
		n_cov = 5
		data = mp.gen_data.gen_panel_data(n_obs=50, n_subjects=20, n_cov=n_cov)
		mpinf_args = dict(times=data['times'], subjects=data['subjects'], clusters=data['subjects'])
		# Compute residuals + LFO residuals
		mpinf = mp.panel.MosaicPanelInference(
			outcomes=data['outcomes'],
			cts_covariates=data['covariates'],
			**mpinf_args,
		)
		mpinf.compute_mosaic_residuals()
		# Loop through several features
		for feature in np.random.choice(n_cov, size=min(n_cov, 5), replace=True):
			# Compute slope/statistic
			mpinf._downdate_mosaic_residuals(feature=feature)
			slope = mpinf.ZA_sums[:, 0].sum()
			stat = mpinf.Zresid_sums[:, 0].sum()
			# Manually leave the feature out and invert the null
			neg_feature = [i for i in range(n_cov) if i != feature]
			bs = np.random.randn(5)
			for b in bs:
				Z = data['covariates'][..., feature]
				new_outcomes = data['outcomes'] - b * Z
				mpinf_manual = mp.panel.MosaicPanelInference(
					outcomes=new_outcomes,
					cts_covariates=data['covariates'][..., neg_feature],
					tile_ids=mpinf.tile_ids,
					**mpinf_args,
				)
				mpinf_manual.compute_mosaic_residuals()
				stat_new = np.sum(mpinf_manual.residuals * Z)
				np.testing.assert_array_almost_equal(
					stat - b * slope,
					stat_new,
					decimal=5,
					err_msg=f"Slope/intercept form used in inversion is inaccurate with b={b}, feature={feature}"
				)

	def test_linear_inversion(self):
		"""
		Tests that the algebra for the linear test statistic inversion is correct.
		"""
		np.random.seed(123)
		nrand, alpha = 100, 0.05
		for _ in range(10):
			slope = 100 * np.random.uniform()
			# Create statistics/null_slopes
			stat = np.random.randn()
			null_slopes = np.random.uniform(-slope, slope, size=nrand)
			null_stats = np.random.randn(nrand)
			# Queries p-value at a specific value of beta
			def query_pval(beta):
				new_stat = np.abs(stat - beta * slope)
				new_null_stats = np.abs(null_stats - beta * null_slopes)
				return (1 + np.sum(new_stat <= new_null_stats)) / (1 + nrand)
			# Compute lower and upper bound
			lower, upper = mp.panel.invert_linear_statistic(
				stat=stat, slope=slope, null_stats=null_stats, 
				null_slopes=null_slopes, alpha=alpha,
			)
			# Query
			tol = 1e-3
			for boundary, bsign, bname in zip(
				[lower, upper], [1, -1], ['lower', 'upper']
			):
				for sign in [-1, 1]:
					beta = boundary + sign * tol
					pval = query_pval(beta)
					self.assertTrue(
						bsign * sign * pval >= bsign * sign * alpha,
						f"invert_linear_statistic produces {bname}={boundary}, but at beta={beta}, pval={pval} with alpha={alpha}"
					)
			for beta in np.random.uniform(lower+tol, upper-tol, size=20):
				pval = query_pval(beta)
				self.assertTrue(
					pval >= alpha,
					f"invert_linear_statistic produces CI={[lower, upper]} but at beta={beta}, pval={pval} with alpha={alpha}"
				)

if __name__ == "__main__":
	# Run tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestTileTransformations()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()
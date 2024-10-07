import numpy as np
import pandas as pd
import scipy.linalg
from typing import Optional, Union
from . import core, utilities, tilings
import sklearn.linear_model as lm
import scipy.sparse
import itertools

def _construct_swapinds(t):
	"""
	Swaps elements t-1 and t of an array of length t.
	"""
	swapinds = np.arange(t)
	swapinds[1::2] -= 1
	swapinds[::2] += 1
	return np.minimum(swapinds, t-1).astype(int)

def construct_tile_transformation(
	subjects: np.array,
	times: np.array,
	covariates: Optional[Union[np.array, scipy.sparse.sparray]]=None,
	invariance='local_exch',
):
	"""
	Constructs a transformation function for a tile of n observations.

	Parameters
	----------
	subjects : np.array
		n-length integer array specifying the subject 
		for each observation.
	times : np.array
		n-length integer array specifying the times
		for each observation.
	covariates : None | np.array | sparse.sparray
		Optional n x p (sparse) array of covariates.
	invariance : str
		A string specifying the type of transformation. Options:

		- 'local_exch': swaps time t and t+1 for all even t.
		- 'local_exch_cts': within each subject, swaps neighboring timepoints.
		- 'local_exch_space': swaps subjects i and i+1 for all odd i. 
		- 'time_reverse': reverses order of each time series
		- 'wild_bs': multiplies all residuals by -1

	Returns
	-------
	transformation : callable
		transformation(residuals) returns the desired
		transformation of the residuals
	"""
	n = len(subjects)
	if len(subjects) != len(times):
		raise ValueError(
			f"different lengths for subjects ({len(subjects)}) and times ({len(times)})"
		)
	# Continuous vs discrete variance
	if invariance == 'local_exch' and np.any(times != times.astype(int)):
		invariance = 'local_exch_cts'
	invariance = str(invariance).lower()
	if invariance in ['local_exch', 'local_exch_cts', 'time_reverse']:
		permutation = np.zeros(n, dtype=int)
		# Loop through subjects
		for subject in np.unique(subjects):
			# time series for this subject
			inds = np.where(subjects == subject)[0]
			if len(inds) == 1:
				permutation[inds] = inds
				continue

			# argsort and swap neighboring observations
			argsort_times = inds[np.argsort(times[inds])]
			rev_inds = np.argsort(np.argsort(times[inds]))
			# Different permutations
			if invariance == 'local_exch_cts':
				subject_swapinds = argsort_times[_construct_swapinds(len(inds))][rev_inds]
			elif invariance == 'local_exch':
				# custom swaps which only swap times t and t+1 for even t; 
				# if time 2 is present but not time 3, then time 2 is not swapped. 
				diffs = times[argsort_times][1:]- times[argsort_times][:-1]
				swapinds = np.arange(len(inds))
				even = times[argsort_times][:-1] % 2 == 0
				swapinds[:-1][(diffs == 1) & even] += 1
				swapinds[1:][(diffs == 1) & even] -= 1
				# then construct
				subject_swapinds = argsort_times[swapinds][rev_inds]
			else:
				subject_swapinds = np.flip(argsort_times)[rev_inds]
			# save
			permutation[inds] = subject_swapinds
		return lambda x: x[permutation]
	if invariance == 'local_exch_space':
		return construct_tile_transformation(
			subjects=times, times=subjects, invariance='local_exch'
		)
	if invariance == 'wild_bs':
		return lambda x: -x
	else:
		raise ValueError(f"Unrecognized invariance={invariance}")

def ols_residuals_panel(
	outcomes: np.array, 
	covariates: np.array,
):	
	"""
	Computes residuals via OLS (NOT cross-sectional OLS).
n
	Parameters
	----------
	outcomes : np.array
		n-length array of outcome data
	covariates : np.array
		(n,p)-shaped array of covariates.

	Returns
	-------
	residuals : np.array
		(n_obs, n_subjects) array of OLS residuals.
	"""
	n_cov = covariates.shape[-1]
	# Case 1: dense.
	if isinstance(covariates, np.ndarray):
		# Flatten for QR decomposition
		A = np.linalg.pinv(covariates.T @ covariates)
		hatbeta = A @ covariates.T @ outcomes
		resid = outcomes - covariates @ hatbeta
	# Case 2: sparse
	elif isinstance(covariates, sparse.COO):
		ols = lm.LinearRegression(fit_intercept=False)
		ols.fit(y=outcomes, X=covariates)
		resid = outcomes - ols.predict(covariates)
	return resid.reshape(*outcomes.shape, order='C')

def _convert_pd_to_numpy(x):
	if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
		return x.values
	return x

class MosaicPanelTest(core.MosaicPermutationTest):
	"""
	Note to self: this is an extension that should go in the appendix IMO.
	"""
	def __init__(
		self,
		outcomes: Union[np.array, pd.Series],
		test_stat: callable,
		subjects: Union[np.array, pd.Series],
		times: Union[np.array, pd.Series],
		# Optional inputs
		clusters: Optional[Union[np.array, pd.Series]]=None,
		cts_covariates: Optional[Union[pd.DataFrame, np.array]]=None,
		discrete_covariates: Optional[pd.DataFrame]=None,
		# tstat kwargs
		tstat_kwargs: Optional[dict]=None,
		# tile / permutation kwargs
		invariance='local_exch',
		ntiles: int=10,
	):
		# Outcomes, subjects, times, and clusters
		self.outcomes = _convert_pd_to_numpy(outcomes)
		self.n = len(self.outcomes)
		self.subjects = tilings._preprocess_partition(_convert_pd_to_numpy(subjects))
		self.times = _convert_pd_to_numpy(times)
		if clusters is not None:
			self.clusters = tilings._preprocess_partition(_convert_pd_to_numpy(clusters))
		else:
			self.clusters = self.subjects
		# Construct covariates
		if cts_covariates is not None:
			self.covariates = _convert_pd_to_numpy(cts_covariates)
		else:
			self.covariates = np.ones((self.n, 1))
		# Possibly add discrete covariates
		if discrete_covariates is not None:
			dummies = mp.utilities.get_dummies_sparse(discrete_covariates)
			self.covariates = scipy.sparse.hstack(
				(scipy.sparse.csr_matrix(self.covariates), dummies)
			)
		self.n_cov = self.covariates.shape[1]
		self.test_stat = test_stat
		self.tstat_kwargs = dict() if tstat_kwargs is None else tstat_kwargs
		# Create tiles: TODO different invariances for creating tiles in the future
		ntiles = int(min(ntiles, len(np.unique(self.clusters))))
		self.tile_ids = tilings.coarsify_partition(self.clusters, k=ntiles, random=False)
		self.ntiles = len(np.unique(self.tile_ids))
		self.tile_inds = [
			np.where(self.tile_ids == tile_id)[0] for tile_id in range(ntiles)
		]
		# Create tile transformations
		self.invariance = invariance
		self.tile_transforms = [
			construct_tile_transformation(
				subjects=self.subjects[self.tile_inds[tile_id]],
				times=self.times[self.tile_inds[tile_id]],
				covariates=self.covariates[self.tile_inds[tile_id]],
				invariance=self.invariance,
			) for tile_id in range(ntiles)
		]

	def compute_mosaic_residuals(self):
		self.residuals = np.zeros(self.n)
		for tile_id, inds in enumerate(self.tile_inds):
			transform = self.tile_transforms[tile_id]
			# augmented covariates: two cases
			if isinstance(self.covariates, np.ndarray):
				aug_cov = np.concatenate([self.covariates[inds], transform(self.covariates[inds])], axis=1)
			else:
				aug_cov = scipy.sparse.hstack((self.covariates[inds], transform(self.covariates[inds])))
			# residuals
			self.residuals[inds] = ols_residuals_panel(
				outcomes=self.outcomes[inds], # this is flat, still fine with this function call
				covariates=aug_cov,
			)
		self._rtilde = self.residuals.copy()

	def permute_residuals(self, _permute_all=False):
		# Note: _permute_all is an internal flag which does all the flips
		# uniforms
		U = np.random.uniform(size=len(self.tile_inds))
		# Loop through and possibly permute
		for tile_id, inds in enumerate(self.tile_inds):
			if U[tile_id] <= 0.5 or _permute_all:
				self._rtilde[inds] = self.tile_transforms[tile_id](self.residuals[inds])
			else:
				self._rtilde[inds] = self.residuals[inds]

class QuadraticMosaicPanelTest(MosaicPanelTest):
	"""
	Parameters
	----------
	method : str
		One of 'cov', 'abscov', 'corr', 'abscorr'
	weights : np.array
		n_subjects-shaped array mapping a subject
		to its weight (which may be negative).
	"""
	def __init__(self, *args, method: str='cov', weights: Optional[np.array]=None, **kwargs):
		super().__init__(*args, **kwargs, test_stat=None)
		self.method = method
		self.weights = weights
		if self.weights is None:
			self.weights = np.ones(len(np.unique(self.subjects)))

	def _compute_p_value(self, nrand: int, verbose: bool) -> float:
		### Preprocessing
		self.permute_residuals(_permute_all=True)
		# Step 1: pivots
		rdf = pd.DataFrame(
			np.stack([self.residuals, self.times, self.subjects], axis=1),
			columns=['residuals', 'times', 'subjects']
		).pivot(
			index='times',
			columns='subjects',
			values='residuals'
		)
		rtildedf = pd.DataFrame(
			np.stack([self._rtilde, self.times, self.subjects], axis=1),
			columns=['residuals', 'times', 'subjects']
		).pivot(
			index='times',
			columns='subjects',
			values='residuals'
		)
		tdf = pd.DataFrame(
			np.stack([self.tile_ids, self.times, self.subjects], axis=1),
			columns=['tiles', 'times', 'subjects']
		).pivot(
			index='times',
			columns='subjects',
			values='tiles'
		)
		# Step 2: create aggregate tile-specific residuals
		weights = pd.Series(self.weights, index=np.sort(np.unique(self.subjects)))
		acr = []
		acrtilde = []
		tilenums = np.unique(self.tile_ids)
		for tilenum in tilenums:
			for l, df in zip([acr, acrtilde], [rdf, rtildedf]):
				l.append((df * weights * (tdf == tilenum)).sum(axis=1))
		acr = pd.DataFrame(acr, index=tilenums).T # T x ntiles
		acrtilde = pd.DataFrame(acrtilde, index=tilenums).T
		# Step 3: Create covariances
		self.all_covs = np.stack(
			[np.stack([acr.values.T @ acr.values, acr.values.T @ acrtilde.values], axis=0),
			np.stack([acrtilde.values.T @ acr.values, acrtilde.values.T @ acrtilde.values], axis=0)],
			axis=0
		)
		# get rid of diagonals
		for z0, z1 in itertools.product([0, 1], [0,1]):
			self.all_covs[z0, z1] -= np.diag(np.diag(self.all_covs[z0, z1]))
		if self.method == 'abscov':
			self.all_covs = np.abs(self.all_covs)
		elif self.method in ['corr', 'abscorr']:
			raise NotImplementedError("This is TODO.")

		# Step 4: create p-value
		self.statistic = np.mean(self.all_covs[0, 0])
		self.null_statistics = np.zeros(nrand).reshape(-1, 1)
		Z = np.random.binomial(1, 0.5, size=(nrand, len(tilenums)))
		for r in utilities.vrange(nrand, verbose=verbose):
			for z0, z1 in itertools.product([0,1], [0,1]):
				self.null_statistics[r] += np.mean(self.all_covs[z0, z1] * np.outer(Z[r]==z0, Z[r]==z1).astype(float))
		self.pval = (1+np.sum(self.statistic <= self.null_statistics)) / (nrand + 1)

		# Compute approximate z-statistic
		comb = np.concatenate([[self.statistic], self.null_statistics.flatten()])
		# Handle when SD == 0
		if comb.std() > 0:
			self.apprx_zstat = (self.statistic - comb.mean()) / comb.std()
		else:
			self.apprx_zstat = 0
		return self.pval
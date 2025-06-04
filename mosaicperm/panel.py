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
		n-length array of OLS residuals.
	"""
	n_cov = covariates.shape[-1]
	# Case 1: dense.
	if isinstance(covariates, np.ndarray):
		# Flatten for QR decomposition
		hatbeta = np.linalg.lstsq(a=covariates, b=outcomes, rcond=None)[0]
		#A = np.linalg.pinv(covariates.T @ covariates)
		#hatbeta = A @ covariates.T @ outcomes
		resid = outcomes - covariates @ hatbeta
	# Case 2: sparse
	elif isinstance(covariates, scipy.sparse.csr_matrix):
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
	Initialize a MosaicPanelTest for panel data analysis.

	Parameters
	----------
	outcomes : np.array or pd.Series
		``n``-length array of outcome data for the panel analysis.
	test_stat : callable
		A function mapping a ``n``-length array of residuals to either:

			- A single statistic measuring evidence against the null.
			- Alternatively, a 1D array of many statistics, in which
				case the p-value will adaptively aggregate evidence across
				all test statistics.

	subjects : np.array or pd.Series
		``n``-length array specifying the subject identifier
		for each observation.
	times : np.array or pd.Series
		``n``-length array specifying the time identifier
		for each observation.
	clusters : np.array, pd.Series, or scipy.sparse.csr_matrix, optional
		``n``-length array where ``clusters[i] = k`` signals that 
		observation i is in the kth cluster. If not provided, 
		defalts to using subjects as clusters.
	cts_covariates : pd.DataFrame or np.array, optional
		``(n, p)``-shaped array of continuous covariates to control for
		in the analysis. If not provided, defaults to an intercept-only model.
	discrete_covariates : pd.DataFrame, optional
		DataFrame of discrete/categorical covariates to be converted
		to dummy variables and included in the model.
	sparse : bool, default True
		If True, uses sparse representation of discrete_covariates.
		Ignored if discrete_covariates is not supplied.
	tstat_kwargs : dict, optional
		Optional kwargs to be passed to ``test_stat``.
	invariance : str, default 'local_exch'
		A string specifying the type of transformation for permutations. Options:

			- 'local_exch': swaps time t and t+1 for all even t.
			- 'local_exch_cts': within each subject, swaps neighboring timepoints.
			- 'time_reverse': reverses order of each time series.
			- 'wild_bs': multiplies all residuals by -1.

	ntiles : int, default 10
		Number of tiles to use for the mosaic permutation test.
	tile_ids : np.array, optional
		``n``-length array where ``tile_ids[i] = k`` implies that observation
		i is in tile k. If not provided, tiles are constructed automatically.
	"""
	def __init__(
		self,
		outcomes: Union[np.array, pd.Series],
		test_stat: callable,
		subjects: Union[np.array, pd.Series],
		times: Union[np.array, pd.Series],
		# Optional inputs
		clusters: Optional[Union[np.array, pd.Series, scipy.sparse.csr_matrix]]=None,
		cts_covariates: Optional[Union[pd.DataFrame, np.array]]=None,
		discrete_covariates: Optional[pd.DataFrame]=None,
		sparse: bool=True,
		# tstat kwargs
		tstat_kwargs: Optional[dict]=None,
		# tile / permutation kwargs
		invariance='local_exch',
		ntiles: int=10,
		tile_ids: Optional[np.array]=None,
	):
		# Outcomes, subjects, times, and clusters
		self.outcomes = _convert_pd_to_numpy(outcomes)
		self.n = len(self.outcomes)
		self.orig_subjects = _convert_pd_to_numpy(subjects)
		self.subjects = tilings._preprocess_partition(self.orig_subjects)
		self.times = _convert_pd_to_numpy(times)
		self.invariance = invariance
		self.ntiles = ntiles
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
			dummies = utilities.get_dummies_sparse(discrete_covariates)
			if sparse:
				self.covariates = scipy.sparse.hstack(
					(scipy.sparse.csr_matrix(self.covariates), dummies)
				)
			else:
				self.covariates = np.concatenate([self.covariates, dummies.toarray()], axis=1)
		self.n_cov = self.covariates.shape[1]
		self.test_stat = test_stat
		self.tstat_kwargs = dict() if tstat_kwargs is None else tstat_kwargs
		self._create_tiles(invariance=invariance, ntiles=ntiles, tile_ids=tile_ids)

	def _create_tiles(self, invariance: str, ntiles: int, tile_ids: Optional[np.array]=None):
		"""
		Creates tiles and tile transformations.
		"""
		# Create tiles
		if tile_ids is None:
			ntiles = int(min(ntiles, len(np.unique(self.clusters))))
			self.tile_ids = tilings.coarsify_partition(self.clusters, k=ntiles, random=False)
		else:
			self.tile_ids = tile_ids
		self.ntiles = len(np.unique(self.tile_ids))
		self.tile_inds = [
			np.where(self.tile_ids == tile_id)[0] for tile_id in range(self.ntiles)
		]
		# Create tile transformations
		self.invariance = invariance
		self.tile_transforms = [
			construct_tile_transformation(
				subjects=self.subjects[self.tile_inds[tile_id]],
				times=self.times[self.tile_inds[tile_id]],
				covariates=self.covariates[self.tile_inds[tile_id]],
				invariance=self.invariance,
			) for tile_id in range(self.ntiles)
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
	Mosaic permutation test for panel data using quadratic test statistics.
	
	This class extends MosaicPanelTest to perform tests based on quadratic forms
	of residual covariances or correlations across tiles. It uses dynamic programming
	to compute the permutation distribution extremely efficiently.

	Parameters
	----------
	*args
		Positional arguments passed to the parent MosaicPanelTest class.
		See :class:`MosaicPanelTest` for details on required arguments
		(outcomes, test_stat, subjects, times, etc.).
	method : str, default 'cov'
		The method for computing the quadratic test statistic. Options:

		- 'cov': Uses covariances between tile-aggregated residuals.
		- 'abscov': Uses absolute values of covariances between tile-aggregated residuals.
		- 'corr': Uses correlations between tile-aggregated residuals.
		- 'abscorr': Uses absolute values of correlations between tile-aggregated residuals.

	weights : np.array or pd.Series, optional
		``n_subjects``-length array mapping each subject to its weight
		in the aggregation. Weights may be negative. If provided as a
		pd.Series, the index should correspond to subject identifiers.
		If not provided, defaults to equal weights of 1 for all subjects.
	**kwargs
		Additional keyword arguments passed to the parent MosaicPanelTest class.
	"""
	def __init__(self, *args, method: str='cov', weights: Optional[np.array]=None, **kwargs):
		super().__init__(*args, **kwargs, test_stat=None)
		self.method = method
		self.weights = weights
		if isinstance(self.weights, pd.Series):
			# Ensure index is compatible with processed subjects
			orig2new = {self.orig_subjects[x]:self.subjects[x] for x in range(len(self.subjects))}
			self.weights.index = self.weights.index.map(orig2new)
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
		self.rdf = rdf
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
		self.tdf = tdf
		# Step 2: create aggregate tile-specific residuals
		weights = pd.Series(self.weights, index=np.sort(np.unique(self.subjects)))
		acr = []
		acrtilde = []
		tilenums = np.unique(self.tile_ids); ntiles = len(tilenums)
		for tilenum in tilenums:
			for l, df in zip([acr, acrtilde], [rdf, rtildedf]):
				l.append((df * weights * (tdf == tilenum)).sum(axis=1))
		acr = pd.DataFrame(acr, index=tilenums).T # T x ntiles
		acrtilde = pd.DataFrame(acrtilde, index=tilenums).T
		self.acr = acr
		self.acrtilde = acrtilde
		# Step 3: Create covariances
		self.all_covs = np.zeros((2, 2, ntiles, ntiles))
		if self.method in ['cov', 'abscov']:
			self.all_covs = np.stack(
				[np.stack([acr.values.T @ acr.values, acr.values.T @ acrtilde.values], axis=0),
				np.stack([acrtilde.values.T @ acr.values, acrtilde.values.T @ acrtilde.values], axis=0)],
				axis=0
			)
		elif self.method in ['corr', 'abscorr']:
			for z0, acr1 in zip([0, 1], [acr.values, acrtilde.values]):
				for z1, acr2 in zip([0, 1], [acr.values, acrtilde.values]):
					self.all_covs[z0, z1] = np.corrcoef(
						acr1.T, acr2.T
					)[0:ntiles][:, ntiles]
		if self.method in ['abscov', 'abscorr']:
			self.all_covs = np.abs(self.all_covs)

		# get rid of diagonals
		for z0, z1 in itertools.product([0, 1], [0,1]):
			self.all_covs[z0, z1] -= np.diag(np.diag(self.all_covs[z0, z1]))

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

def _precompute_lfo_qr(X, transform, lmda=1e-8):
	"""
	Computes QR of augmented design with regularization to ensure
	numerical stability for low-rank designs.
	"""
	# Normalize
	Xaug = np.concatenate([X, transform(X)], axis=1)
	ses = Xaug.std(axis=0)
	ses[ses == 0] = 1
	Xaug = Xaug / ses
	# Add
	Xaug = np.concatenate([Xaug, lmda * np.eye(Xaug.shape[1])], axis=0)
	# QR
	return np.linalg.qr(Xaug)

def _compute_lfo_resid(
	fullQ: np.array,
	fullR: np.array,
	outcomes: np.array,
	feature: int,
	n_cov: int,
	Z: Optional[np.array]=None
):
	"""
	Uses precomputed Q, R of full design to compute leave-feature-out
	residuals.

	Parameters
	----------
	fullQ : np.array
		(n + 2*n_cov) x (2* n_cov) Q from QR decomp of full augmented matrix.
	fullR : np.array
		(2*n_cov x 2*n_cov) R from QR decomp of full augmented matrix
	outcomes : np.array
		array of outcomes
	feature : int
		Integer between 0 and n_cov - 1 specifying which feature is of interest.
	n_cov : int
		Number of covariates
	Z : np.array
		n-length array of values taken by feature of interest, optional.

	Returns
	-------
	resid : np.array
		Residuals of ``outcomes" after projecting out the augmented design matrix
		without the ``feature``.
	Zresid : np.array
		Residuals of ``Z``, assuming ``Z`` is provided.
	"""
	# Update
	Q = fullQ.copy()
	R = fullR.copy()
	for k in [feature + n_cov, feature]:
		Q, R = scipy.linalg.qr_delete(Q, R, k=k, p=1, which='col', check_finite=True)
	# Get rid of last 2*n_cov rows of Q which are for numerical stability
	Qsub = Q[0:len(outcomes)]
	resid =  outcomes - Qsub @ (Qsub.T @ outcomes)
	if Z is None:
		return resid
	else:
		Zresid = Z - Qsub @ (Qsub.T @ Z)
		return resid, Zresid


class MosaicPanelInference(MosaicPanelTest):
	"""
	Mosaic permutation-based inference for linear models in panel data.
	
	Parameters
	----------
	*args
		Positional arguments passed to the parent MosaicPanelTest class.
		See :class:`MosaicPanelTest` for details on required arguments
		(outcomes, subjects, times, etc.). Note that ``test_stat`` is
		automatically set to None as it is not used in this inference class.
	**kwargs
		Additional keyword arguments passed to the parent MosaicPanelTest class.
		Common arguments include ``cts_covariates``, ``discrete_covariates``,
		``clusters``, ``invariance``, ``ntiles``, etc.

	Notes
	-----
	For dense covariate matrices, the class uses efficient QR decomposition
	updates to avoid recomputing regressions for each feature. For sparse
	matrices, it falls back to recomputing regressions as needed.

	Examples
	--------
	>>> import numpy as np
	>>> import pandas as pd
	>>> import mosaicperm as mp
	>>> 
	>>> # Generate synthetic panel data
	>>> n_subjects, n_times = 50, 20
	>>> n_obs = n_subjects * n_times
	>>> subjects = np.repeat(np.arange(n_subjects), n_times)
	>>> times = np.tile(np.arange(n_times), n_subjects)
	>>> 
	>>> # Generate covariates and outcomes
	>>> X = np.random.randn(n_obs, 3)
	>>> beta_true = np.array([1.0, -0.5, 0.2])
	>>> outcomes = X @ beta_true + np.random.randn(n_obs) * 0.5
	>>> 
	>>> # Fit mosaic panel inference
	>>> mpi = mp.panel.MosaicPanelInference(
	...     outcomes=outcomes,
	...     subjects=subjects,
	...     times=times,
	...     cts_covariates=X,
	...     ntiles=8
	... )
	>>> mpi.fit(nrand=1000, alpha=0.05)
	>>> print(mpi.summary)
	
	The summary will show estimates, standard errors, confidence intervals,
	and p-values for each coefficient.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs, test_stat=None)

	def compute_mosaic_residuals(self, verbose: bool=True):
		"""
		Computes full mosaic residuals and precomputes useful quantities.
		"""
		if isinstance(self.covariates, np.ndarray):
			# Loop through tiles
			self.residuals = np.zeros(self.n)
			self._tile_QRs = dict() # maps tile_num --> (Q, R) of augmented, regularized design
			for tile_id in utilities.vrange(len(self.tile_inds), verbose=verbose):
				inds = self.tile_inds[tile_id]
				# Compute Q, R decomp of augmented design
				Q, R = _precompute_lfo_qr(self.covariates[inds], transform=self.tile_transforms[tile_id])
				self._tile_QRs[tile_id] = Q, R
				# Compute final residuals
				Qsub = Q[0:len(inds)]
				self.residuals[inds] = self.outcomes[inds] - Qsub @ (Qsub.T @ self.outcomes[inds])
			# Initialize rtilde correctly
			self._rtilde = self.residuals.copy()
			return self.residuals
		else:
			super().compute_mosaic_residuals()

	def _downdate_mosaic_residuals(self, feature: int) -> np.array:
		"""
		Efficiently computes mosaic residuals leaving 
		out feature j.

		Parameters
		----------
		feature : int
			Integer in {0, ..., n_cov - 1}.

		Returns
		-------
		lfo_resids : np.array
			Resids leaving out ``feature``

		Notes
		-----
		Also projects out the influence of the other covariates on
		``self.covariates[:, :, feature]`` and stores the result in
		``self.orthog_feature``.
		"""
		# Stores "leave feature out" residuals
		self.lfo_resids = np.zeros(self.residuals.shape)
		# Stores the feature
		self.orthog_feature = np.zeros(self.residuals.shape)
		# If Z is the feature and A is the orthogonalized feature, 
		# stores np.sum(Z * A) and np.sum(Z * transform(A)), etc.
		self.Zresid_sums = np.zeros((self.ntiles, 2))
		self.ZA_sums = np.zeros((self.ntiles, 2))
		for tile_id, inds in enumerate(self.tile_inds):
			Z = self.covariates[inds, feature]
			transform = self.tile_transforms[tile_id]
			# In the non-sparse case, use efficient leave-one-out updates
			if isinstance(self.covariates, np.ndarray):
				# Downdate QR and obtain residuals
				fullQ, fullR = self._tile_QRs[tile_id]
				resid, A = _compute_lfo_resid(
					fullQ=fullQ, 
					fullR=fullR, 
					outcomes=self.outcomes[inds], 
					feature=feature, 
					n_cov=self.n_cov,
					Z=Z
				)
			# In the sparse case, just redo each regression
			else:
				Z = Z.toarray().flatten()
				neg_feature = np.array([j for j in range(self.n_cov) if j != feature]).astype(int)
				# augmented covariates
				Xtile = self.covariates[inds][:, neg_feature]
				aug_cov = scipy.sparse.hstack([Xtile, transform(Xtile)])
				# residuals
				resid = ols_residuals_panel(
					outcomes=self.outcomes[inds],
					covariates=aug_cov,
				)
				A = ols_residuals_panel(
					outcomes=Z,
					covariates=aug_cov,
				)
			# Save
			self.lfo_resids[inds] = resid
			self.orthog_feature[inds] = A
			# Precompute sums used to compute slopes/intercepts within tiles
			# Note that A @ resid = Z @ resid and Z @ A = A @ A
			# and using A instead of Z is slightly more stable
			self.Zresid_sums[tile_id, 0] = np.sum(A * resid)
			self.Zresid_sums[tile_id, 1] = np.sum(A * transform(resid))
			self.ZA_sums[tile_id, 0] = np.sum(A * A)
			self.ZA_sums[tile_id, 1] = np.sum(A * transform(A))
		# return
		return self.lfo_resids


	def _precompute_confidence_intervals(
		self, 
		nrand: int,
		features: Optional[np.array]=None,
		verbose: bool=True,
		use_centered_stat: bool=False,
	):
		"""
		Computes underlying confidence intervals for :meth:`fit`.

		Parameters
		----------
		alpha : int
			Desired nominal Type I error rate.
		nrand : int
			Number of randomizations.
		features : np.array
			The list of features to compute CIs for. 
			Defaults to all features.
		verbose : bool
			If True, show progress bars.
		"""
		# process features
		if features is None:
			features = np.arange(self.n_cov).astype(int)
		if not utilities.haslength(features):
			features = np.array([features]).astype(int)
		self.features = features

		# Save
		self.estimates = np.zeros(self.n_cov); self.estimates[:] = np.nan
		self.ses = self.estimates.copy()
		self.lcis = self.ses.copy()
		self.ucis = self.ses.copy()
		self.pvals = self.ses.copy()

		# Save the slopes and statistics
		self._stats = np.zeros(self.n_cov); self._stats[:] = np.nan
		self._slopes = self._stats.copy()
		# save null slopes/statistics
		self._null_stats = np.zeros((self.n_cov, nrand))
		self._null_stats[:] = np.nan
		self._null_slopes = self._null_stats.copy()

		# Randomization (common to all features)
		flips = np.random.binomial(1, 0.5, size=(nrand, self.ntiles)).astype(int)
		# Loop through and compute
		xinds = np.stack([np.arange(self.ntiles) for _ in range(nrand)])
		for j in utilities.vrange(len(features), verbose=verbose):
			feature = features[j]
			self._downdate_mosaic_residuals(feature=feature)
			if use_centered_stat:
				# Observed test statistic + slope as beta changes
				stat = self.Zresid_sums[:, 0].sum() - self.Zresid_sums[:, 1].sum()
				slope = self.ZA_sums[:, 0].sum() - self.ZA_sums[:, 1].sum()
				# Compute null test statistics + slopes as beta changes
				null_stats = (self.Zresid_sums[(xinds, flips)] - self.Zresid_sums[(xinds, 1-flips)]).sum(axis=1)
				null_slopes = (self.Zresid_sums[(xinds, flips)] - self.Zresid_sums[(xinds, 1-flips)]).sum(axis=1)
			else:
				# Observe test statistic + slope as beta changes
				stat = self.Zresid_sums[:, 0].sum()
				slope = self.ZA_sums[:, 0].sum()
				# Compute null test statistics + slopes as beta changes
				null_stats = self.Zresid_sums[(xinds, flips)].sum(axis=1)
				null_slopes = self.ZA_sums[(xinds, flips)].sum(axis=1)
			# Save
			self._null_stats[j] = null_stats
			self._null_slopes[j] = null_slopes
			self._stats[j] = stat
			self._slopes[j] = slope
			# Compute p-value
			self.pvals[j] = (1 + np.sum(np.abs(stat) <= np.abs(null_stats))) / (1 + nrand)
			# # Estimate
			self.estimates[j] = stat / slope
			# # null estimators under the null where beta_j = estimate
			null_estimates = (null_stats - self.estimates[j] * null_slopes) / slope
			self.ses[j] = np.sqrt(np.mean(null_estimates**2))

	def compute_cis(self, alpha=0.05):
		for j in self.features:
			self.lcis[j], self.ucis[j] = invert_linear_statistic(
				stat=self._stats[j],
				slope=self._slopes[j],
				null_stats=self._null_stats[j],
				null_slopes=self._null_slopes[j],
				alpha=alpha
			)
		# To save the confidence intervals
		columns = ['Estimate', 'SE', 'Lower', 'Upper', 'p-value']
		self._summary = pd.DataFrame(
			np.stack(
				[self.estimates, self.ses, self.lcis, self.ucis, self.pvals], axis=1
			),
			index=np.arange(self.n_cov),
			columns=columns,
		)
		self._summary = self._summary.loc[self._summary['Estimate'].notnull()]


	def fit(
		self,
		nrand: int=10000,
		alpha: int=0.05,
		features: Optional[np.array]=None,
		verbose: bool=True,
		use_centered_stat: bool=False,
	):
		"""
		Fits the linear model and computes confidence interva

		Parameters
		----------
		nrand : int
			Number of randomizations.
		alpha : int
			Desired nominal Type I error rate.
		features : np.array
			The list of features to compute CIs for. 
			Defaults to all features.
		verbose : bool
			If True, show progress bars.
		use_centered_stat : bool
			If True, uses a randomization-centered statistic.
			This (perhaps pardoxically) leads to slightly higher
			standard errors but narrower confidence intervals. 

		Returns
		-------
		self : object
		"""
		if verbose:
			print("Computing mosaic residuals.")
		#self.compute_mosaic_residuals(verbose=verbose)
		self.compute_mosaic_residuals(verbose=verbose)
		if verbose:
			print("Precomputing confidence intervals.")
		self._precompute_confidence_intervals(
			nrand=nrand,
			features=features,
			verbose=verbose, 
			use_centered_stat=use_centered_stat
		)
		if verbose:
			print("Computing CIs.")
		self.compute_cis(alpha=alpha)
		return self

	@property
	def summary(self):
		"""
		Produces a summary of key inferential results.
		"""
		return self._summary

### Confidence intervals
def invert_linear_statistic(
	stat: float, 
	slope: float,
	null_stats: np.array,
	null_slopes: np.array,
	alpha: float,
	tol: float=1e-8
):
	"""
	Finds the set of beta such that
	
	|stat - beta slope| >= Q_{alpha}(|null_stats - beta * null_slopes|)
	
	under the assumption that |null_slopes| <= |slope|.
	"""
	nrand = len(null_stats)
	if slope < -tol:
		raise ValueError(f"slope={slope} <= 0")
	elif np.abs(slope) < tol:
		return -np.inf, np.inf
	# Adjust absolute values to account for precision errors
	to_adjust = np.abs(null_slopes) > slope
	null_slopes[to_adjust] = slope * np.sign(null_slopes[to_adjust])
	# Equi-tailed threshold
	rthresh = int(np.floor(alpha * (nrand + 1))) - 1
	if rthresh < 0:
		return -np.inf, np.inf
	# First set of inflections
	inflections0 = np.zeros(nrand) - np.inf
	denoms0 = slope - null_slopes
	flags0 = denoms0 != 0
	inflections0[flags0] = (stat - null_stats)[flags0] / denoms0[flags0]
	# Second set of inflections
	inflections1 = np.zeros(nrand) + np.inf
	denoms1 = slope + null_slopes
	flags1 = denoms1 != 0
	inflections1[flags1] = (stat + null_stats)[flags1] / denoms1[flags1]
	# Combine and sort
	inflections = np.sort(np.stack([inflections0, inflections1], axis=1), axis=1)
	# Confidence bounds
	lci = np.sort(inflections[:, 0])[rthresh]
	uci = np.sort(inflections[:, 1])[-(rthresh+1)]
	return lci, uci

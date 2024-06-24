import numpy as np
import pandas as pd
from . import tilings, utilities, core, statistics
from typing import Optional

def ols_residuals(
	outcomes: np.array,
	exposures: np.array,
):
	"""
	Computes residuals via cross-sectional OLS.

	Parameters
	----------
	outcomes : np.array
		(n_obs, n_subjects) array of outcome data.
	exposures : np.array
		(n_obs, n_subjects, n_factors) array of factor exposures
		OR 
		(n_subjects, n_factors) array of factor exposures if
		the exposures do not change with time.
	
	Returns
	-------
	residuals : np.array
		(n_obs, n_subjects) array of residuals from OLS
		cross-sectional regressions.
	"""
	# Case 1: exposures do not change with time
	if len(exposures.shape) == 2:
		Q, _ = np.linalg.qr(exposures)
		return outcomes - (Q @ (Q.T @ outcomes.T)).T
	# Case 2: exposures change with time
	elif len(exposures.shape) == 3:
		residuals = np.zeros(outcomes.shape)
		for i in range(len(residuals)):
			Q, _ = np.linalg.qr(exposures[i])
			residuals[i] = outcomes[i] - Q @ (Q.T @ outcomes[i])
		return residuals
	else:
		raise ValueError(f"Exposures must be a 2D or 3D array, but has shape {exposures.shape}")

_mosaic_factor_data_doc = """
	outcomes : np.array
		(``n_obs``, ``n_subjects``) array of outcomes, e.g., asset returns.
		``outcomes`` may contain nans to indicate missing values.
	exposures : np.array
		(``n_obs``, ``n_subjects``, ``n_factors``) array of factor exposures
		OR 
		(``n_subjects``, ``n_factors``) array of factor exposures if
		the exposures do not change with time.
"""


class MosaicFactorTest(core.MosaicPermutationTest):
	__doc__ = """
	Mosaic test for factor models with known exposures.

	Parameters
	----------
	{data_doc}
	test_stat : function
		A function mapping a (``n_obs``, ``n_subjects``)-size 
		array of residuals to either:

		- A single statistic measuring evidence against the null.
		- Alternatively, a 1D array of many statistics, in which
		  case the p-value will adaptively aggregate evidence across
		  all test statistics.

	test_stat_kwargs : dict
		Optional kwargs to be passed to ``test_stat``.
	tiles : mosaicperm.tilings.Tiling
		An optional :class:`.Tiling` to use as the tiling.
	clusters : np.array
		An optional ``n_subject``-length array where 
		``clusters[i] = k`` signals that subject i is in the
		kth cluster. If ``tiles`` is not provided, this argument
		allows one to test the null that the residuals are independent
		between clusters but possibly dependent within clusters.
		If ``tiles`` is  provided, this argument is ignored.
	**kwargs : dict
		Optional kwargs to :func:`.default_factor_tiles`.
		Ignored if ``tiles`` is provided.

	Examples
	--------
	We run a mosaic permutation test on synthetic data:

	>>> import numpy as np
	>>> import mosaicperm as mp
	>>> 
	>>> # synthetic outcomes and exposures
	>>> n_obs, n_subjects, n_factors = 100, 200, 20
	>>> outcomes = np.random.randn(n_obs, n_subjects)
	>>> exposures = np.random.randn(n_obs, n_subjects, n_factors)
	>>> 
	>>> # example of missing data
	>>> outcomes[0:10][:, 0:5] = np.nan
	>>> exposures[0:10][:, 0:5] = np.nan
	>>> 
	>>> # fit mosaic permutation test
	>>> mpt = mp.factor.MosaicFactorTest(
	... 	outcomes=outcomes,
	... 	exposures=exposures,
	... 	test_stat=mp.statistics.mean_maxcorr_stat,
	>>> )
	>>> mpt.fit().summary()

	We can also produce a time series plot of this analysis:

	>>> mpt.fit_tseries(nrand=100, n_timepoints=20)
	>>> fig, ax = mpt.plot_tseries()

	""".format(data_doc=_mosaic_factor_data_doc)

	def __init__(
		self,
		outcomes: np.array,
		exposures: np.array,
		test_stat: callable,
		test_stat_kwargs: Optional[dict]=None,
		tiles: Optional[list]=None, 
		clusters: Optional[np.array]=None,
		**kwargs,
	):
		# Data
		if isinstance(outcomes, pd.DataFrame):
			outcomes = outcomes.values
		if isinstance(exposures, pd.DataFrame):
			exposures = exposures.values
		self.outcomes = outcomes.copy()
		self.exposures = exposures.copy()
		# Remove nans
		self.n_obs, self.n_subjects = outcomes.shape
		missing_pattern = np.isnan(self.outcomes)
		if np.any(missing_pattern):
			# in this case, must make exposures 3-dimensional since the nan
			# pattern will causes the exposures to change with time
			if len(self.exposures.shape) == 2:
				self.exposures = np.stack(
					[self.exposures for _ in range(self.n_obs)], axis=0
				)
			# fill with zeros (provably preserving validity)
			self.exposures[missing_pattern] = 0
			self.outcomes[missing_pattern] = 0
		# fill additional missing exposures with zero
		self.exposures[np.isnan(self.exposures)] = 0
		# Remove factors with all zero exposures
		if len(self.exposures.shape) == 3:
			to_keep = np.any(self.exposures != 0, axis=(0,1))
		else:
			to_keep = np.any(self.exposures != 0, axis=0)
		self.exposures = self.exposures[..., to_keep]
		# Test statistic
		self.test_stat = test_stat
		self.tstat_kwargs = test_stat_kwargs
		if self.tstat_kwargs is None:
			self.tstat_kwargs = {}
		# Tiles
		self.clusters = clusters
		self.tiles = tiles
		if self.tiles is None:
			self.tiles = tilings.default_factor_tiles(
				exposures=self.exposures,
				n_obs=len(self.outcomes),
				clusters=self.clusters,
				**kwargs
			)
		# Readjust outcomes to ensure that missing pattern
		# does not yield to local exchangeability violations
		for (batch, group) in self.tiles:
			missing_subjects = np.any(missing_pattern[np.ix_(batch, group)], axis=0)
			self.outcomes[np.ix_(batch, group[missing_subjects])] = 0

		# initialize
		super().__init__()

	def compute_mosaic_residuals(self) -> np.array:
		self.residuals = np.zeros(self.outcomes.shape)
		for tile in self.tiles:
			batch, group = tile
			# tile-specific loadings
			# Case 1: loadings do not change with time
			if len(self.exposures.shape) == 2:
				Ltile = self.exposures[group]
			# Case 2: loadings do change with time
			else:
				# Check exposures are constant within tile. Else, adjust
				constant = np.all([
					np.all(self.exposures[j, group] == self.exposures[batch[0], group]) 
					for j in batch
				])
				if constant:
					Ltile = self.exposures[batch[0], group]
				else:
					Ltile = np.concatenate(
						[self.exposures[j, group] for j in batch],
						axis=1
					)
					# for computational efficiency, get rid of duplicate columns
					Ltile = np.unique(Ltile, axis=1)

			# Compute residuals
			Ytile = self.outcomes[np.ix_(batch, group)] # len(batch) x len(group)
			Q, _ = np.linalg.qr(Ltile) # Q is len(group) x n_factors
			self.residuals[np.ix_(batch, group)] = (Ytile.T - Q @ (Q.T @ Ytile.T)).T

		# initialize null permutation object
		self._rtilde = self.residuals.copy()
		return self.residuals


def _window_sum(x: np.array, window: Optional[int]):
	"""
	Returns np.cumsum(x) if window is None. 
	Else, returns np.convolve(x, np.ones(window), 'valid').
	"""
	if window is None:
		return np.cumsum(x)
	else:
		return np.convolve(x, np.ones(window), 'valid')

class MosaicBCV(MosaicFactorTest):
	__doc__ = """
	Mosaic factor test based on :func:`mosaicperm.statistics.adaptive_mosaic_bcv_stat`

	Parameters
	----------
	{data_doc}
	new_exposures : np.array or list
		(``n_models``, ``n_subject``)-shaped array 
		such that ``new_exposures[i]`` is an array
		of new exposures.		
	tiles : mosaicperm.tilings.Tiling
		An optional :class:`.Tiling` to use as the tiling.
	**kwargs : dict
		Optional kwargs to :func:`.default_factor_tiles`.
		Ignored if ``tiles`` is provided.

	Examples
	--------

	>>> import numpy as np
	>>> import mosaicperm as mp
	>>> 
	>>> # synthetic outcomes and exposures
	>>> n_obs, n_subjects, n_factors = 100, 200, 20
	>>> outcomes = np.random.randn(n_obs, n_subjects)
	>>> exposures = np.random.randn(n_obs, n_subjects, n_factors)
	>>> 
	>>> # estimate candidate exposures on the first half of the data
	>>> n0 = int(0.5 * n_obs)
	>>> resid0 = mp.factor.ols_residuals(outcomes[0:n0], exposures[0:n0])
	>>> new_exposures = mp.statistics.approximate_sparse_pcas(np.corrcoef(resid0.T))
	>>> 
	>>> # fit mosaic permutation test
	>>> mpt_bcv = mp.factor.MosaicFactorTest(
	... 	outcomes=outcomes[n0:],
	... 	exposures=exposures[n0:],
	... 	new_exposures=new_exposures,
	>>> )
	>>> mpt_bcv.fit().summary()

	""".format(
		data_doc=_mosaic_factor_data_doc,
	)

	def __init__(
		self,
		new_exposures: np.array,
		*args, 
		**kwargs, 
	):
		super().__init__(
			*args, **kwargs, test_stat=None,
		)
		# Ensure a consistent input format
		if len(new_exposures.shape) == 1:
			new_exposures = new_exposures.reshape(1, -1)
		self.new_exposures = new_exposures

		# test statistic and kwargs
		self.test_stat = statistics.adaptive_mosaic_bcv_stat
		self.tstat_kwargs['tiles'] = self.tiles
		self.tstat_kwargs['new_exposures'] = self.new_exposures

	def summary(self, *args, **kwargs):
		if 'coordinate_index' not in kwargs:
			kwargs['coordinate_index'] = [
				fr"$R^2$ for candidate model {i}" for i in range(len(self.new_exposures))
			]
		return super().summary(*args, **kwargs)


	def _compute_p_value_tseries(
		self, nrand: int, verbose: bool, n_timepoints: int, window: Optional[int],
	):
		# Note:
		# The docs/signature are inherited from the core MosaicFactorTest class.
		# oos_r2s
		new_exposures = self.tstat_kwargs['new_exposures']
		tiles = self.tstat_kwargs['tiles']
		mus = self.tstat_kwargs.get("mus", None)
		if mus is None:
			mus = self.residuals.mean(axis=0)

		n_models = len(new_exposures)
		active = np.any(new_exposures != 0, axis=0) # actuve features
		baseline = _window_sum(np.sum(self.residuals[:, active]**2, axis=1), window=window)
		
		_stats = np.zeros((len(baseline), nrand+1, n_models))
		# Initialize so _stats[:, 0] is the true statistic. 
		self._rtilde = self.residuals.copy()
		for r in utilities.vrange(nrand+1, verbose=verbose):
			for i in range(n_models):
				oos_resids = statistics._bcv_oos_resids(
					self._rtilde, new_exposure=new_exposures[i], tiles=tiles, mus=mus
				)
				oos_l2s = np.sum(oos_resids[:, active]**2, axis=1) # n_obs length array
				oos_errors = _window_sum(oos_l2s, window=window)
				_stats[:, r, i] = 1 - oos_errors / baseline
			# Permute
			self.permute_residuals()

		# compute adaptive p-value and put everything in the right place
		self.null_tseries = _stats[:, 1:]
		self.stats_tseries = _stats[:, 0]
		self.adapt_stats_tseries = np.zeros(len(baseline))
		self.null_adapt_tseries = np.zeros((len(baseline), nrand))
		self.pval_tseries = np.zeros(len(baseline))
		for i in range(len(baseline)):
			out = core.compute_adaptive_pval(
				self.stats_tseries[i], self.null_tseries[i]
			)
			self.pval_tseries[i] = out[0]
			self.adapt_stats_tseries[i] = out[1]
			self.null_adapt_tseries[i] = out[2]

		# z-statistics
		self._compute_apprx_zstat_tseries()

		# create self.starts/self.ends to signal indices of output
		if window is not None:
			self.starts = np.arange(0, self.n_obs - window + 1)
			self.ends = np.arange(window, self.n_obs + 1)
		else:
			self.starts = np.zeros(self.n_obs)
			self.ends = np.arange(1, self.n_obs+1)
import abc
import numpy as np
import pandas as pd
from scipy import stats
from . import utilities
from typing import Optional

def compute_adaptive_pval(
	statistic: np.array, 
	null_statistics: np.array,
) -> tuple:
	"""
	Computes an adaptive p-value based on one or more test statistics.

	Parameters
	----------
	statistic : float or np.array
		Either a float (one statistic) or 1D array (many test statistics).
	null_statistics : np.array
		(``nrand``, ``d``)-shaped array of null statistics computed based on
		null permutations, where ``d`` is the dimension of ``statistic``.
	
	Returns
	-------
	pval : float
		The adaptive p-value.
	adapt_statistic : float
		The adaptive statistic value.
	null_adapt_statistics : np.array
		nrand-shaped array of null adaptive statistic values.
	"""
	# Dimensions
	nrand, d = null_statistics.shape
	if not utilities.haslength(statistic):
		statistic = np.array([statistic])
	# concatenate and take mean/sd
	all_stats = np.concatenate(
		[statistic.reshape(1, d), null_statistics], axis=0
	)
	mu0, sd0 = all_stats.mean(axis=0), all_stats.std(axis=0)
	sd0[sd0 == 0] = 1
	# t-statistic and take maximum
	tstats = (all_stats - mu0) / sd0
	adapt_stats = np.max(tstats, axis=1)
	# return pval
	pval = np.sum(adapt_stats[0] <= adapt_stats) / (nrand + 1)
	return pval, adapt_stats[0], adapt_stats[1:]


def _create_tseries_plot(
	xvals: np.array, # x-axix
	ystat: np.array, # statistic values
	pvals: np.array, # p-values
	zapprx: np.array, # approximate Z-statistics
	ynull_mean: Optional[np.array]=None, # null means
	yquant: Optional[np.array]=None, # null quantiles
	alpha: float=0.05, # threshold
	show_plot: bool=True,
	**figure_kwargs
):
	"""
	Private function which is used to create tseries plots.
	"""
	import matplotlib.pyplot as plt
	figure_kwargs['figsize'] = figure_kwargs.get("figsize", (9, 7)) # default
	
	# Create subplots
	fig = plt.figure(**figure_kwargs)
	ax0 = plt.subplot(2, 1, 1)
	ax1 = plt.subplot(2, 2, 3)
	ax2 = plt.subplot(2, 2, 4)
	axes = [ax0, ax1, ax2]
	
	## First plot: statistic value
	# Plot statistic value
	axes[0].plot(xvals, ystat, color='cornflowerblue', label='Mosaic test statistic')
	axes[0].scatter(xvals, ystat, color='cornflowerblue')
	# Possibly add null mean
	if ynull_mean is not None:
		axes[0].plot(xvals, ynull_mean, color='orangered', label='Null mean')
		axes[0].scatter(xvals, ynull_mean, color='orangered')
	# Possibly add quantile
	if yquant is not None:
		axes[0].plot(
			xvals, 
			yquant, 
			color='orangered',
			label=rf'Null quantile, $\alpha$={alpha}',
			linestyle='dotted',
		)
		axes[0].scatter(xvals, yquant, color='orangered')
		axes[0].set(title="Statistic value and threshold")
	else:
		axes[0].set(title='Statistic value')
	axes[0].set(xlabel='Time', ylabel='Statistic value')
	axes[0].legend()
	
	## Subplot 1: p-value
	zvals = stats.norm.ppf(1-pvals)
	zvals[zvals == - np.inf] = zvals[zvals > - np.inf].min()
	axes[1].plot(xvals, zvals, color='blue', label='Observed')
	axes[1].scatter(xvals, zvals, color='blue')
	axes[1].axhline(
		stats.norm.ppf(1-alpha),
		color='black',
		linestyle='dotted',
		label=rf'Threshold ($\alpha$={alpha})'
	)
	axes[1].set(xlabel='Time', ylabel=r'Z-statistic: $\Phi(1-p)$')
	axes[1].set(title="Exact z-statistic")
	axes[1].legend()
	axes[1].set_ylim(min(0, zvals.min()-zvals.std()/20))    
	
	## Subplot 2: approximate z-statistic
	axes[2].plot(xvals, zapprx, color='blue', label='Observed')
	axes[2].scatter(xvals, zapprx, color='blue')
	axes[2].axhline(
		stats.norm.ppf(1-alpha),
		color='black',
		linestyle='dotted',
		label=rf'Threshold ($\alpha$={alpha})'
	)
	axes[2].set(xlabel='Time', ylabel=r'(S - E[S]) / sd(S)')
	axes[2].set(title="Approximate z-statistic")
	axes[2].set_ylim(min(0, zapprx.min()-zapprx.std()/20))
	axes[2].legend()
	# Adjust
	plt.subplots_adjust(hspace=0.2)
	#return
	if show_plot:
		plt.show()
	else:
		return fig, axes

class MosaicPermutationTest(abc.ABC):
	"""
	Generic class meant for subclassing.
	"""
	def __init__(self):
		self._precompute_permutation_helpers()

	def _precompute_permutation_helpers(self):
		"""
		Precomputes some matrices to ensure faster tile permutations.
		"""
		# maps i, j to the tile number
		self._tilenums = np.empty(self.outcomes.shape, dtype=int)
		# sum of batch lengths
		self._batchlen_sum = np.sum([len(batch) for batch, _ in self.tiles])
		# self._batchinds[i, j] maps coordinate (i, j) to a unique coordinate
		# from 0 to self.batchlen_sum depending only on i and the tile of i,j
		self._batchinds = np.zeros(self.outcomes.shape, dtype=int)
		# loop through and fill
		counter = 0
		for i, (batch, group) in enumerate(self.tiles):
			br = batch.reshape(-1, 1)
			gr = group.reshape(1, -1)
			self._tilenums[br, gr] = i
			self._batchinds[br, gr] = np.arange(counter, counter+len(batch)).reshape(-1, 1)
			counter += len(batch)

	@abc.abstractmethod
	def compute_mosaic_residuals(self):
		"""
		Computes mosaic-style residual estimates.
		
		Returns
		-------
		residuals : np.array
			(``n_obs``, ``n_subjects``)-shaped array of residual estimates.
		"""
		raise NotImplementedError()

	def permute_residuals(
		self, 
		method: str='argsort'
	) -> None:
		"""
		Permutes residuals within tiles. Result is stored in ``self._rtilde``.

		Parameters
		----------
		method : str

			- 'ix' is naive and has complexity O(``n_subjects`` * ``n_obs``) but uses
			  a for loop and can be slow in practice.
			- 'argsort' has complexity O(``n_obs`` log(``n_obs``) * ``n_subjects``) but 
			  is faster in practice.

		Returns
		-------
		None : NoneType
		"""
		# Compute null variant
		if method == 'argsort':
			self._rtilde = np.take_along_axis(
				self.residuals,
				np.argsort(
					self._tilenums + np.random.uniform(size=self._batchlen_sum)[self._batchinds], 
					axis=0,
				),
				axis=0
			)
		else:
			for (batch, group) in self.tiles:
				# Permute within tile
				# this is faster than using np.ix_
				br_shuffle = np.random.permutation(batch).reshape(-1, 1)
				br = batch.reshape(-1, 1)
				gr = group.reshape(1, -1)
				self._rtilde[br, gr] = self.residuals[br_shuffle, gr]

	def _compute_p_value(self, nrand: int, verbose: bool) -> float:
		"""
		Produces the underlying mosaic p-value in :meth:`fit`.

		Parameters
		----------
		nrand : int
			Number of randomizations to perform.
		verbose : bool
			If True (default), displays a progress bar. 

		Returns
		-------
		pval : float
		"""
		# Compute statistic and infer its dimension (for adaptive statistics)
		self.statistic = self.test_stat(self.residuals, **self.tstat_kwargs)
		if utilities.haslength(self.statistic):
			d = len(self.statistic)
		else:
			d = 1
		# compute null statistics
		self.null_statistics = np.zeros((nrand, d))
		for r in utilities.vrange(nrand, verbose=verbose):
			# Compute null statistic
			self.permute_residuals()
			self.null_statistics[r] = self.test_stat(self._rtilde, **self.tstat_kwargs)

		# compute p-value
		out = compute_adaptive_pval(
			self.statistic, self.null_statistics
		)
		self.pval, self.adapt_stat, self.null_adapt_stats = out
		
		# Compute approximate z-statistic
		# Use adaptive vs. original statistic
		if d == 1:
			nstats = self.null_statistics
			stat = self.statistic
		else:
			stat = self.adapt_stat
			nstats = self.null_adapt_stats
		# Handle when SD == 0
		if nstats.std() > 0:
			self.apprx_zstat = (stat - nstats.mean()) / nstats.std()
		else:
			self.apprx_zstat = np.nan
		
		# return p-value
		return self.pval

	def fit(
		self,
		nrand: int=500,
		verbose: bool=True,
	):
		"""
		Runs the mosaic permutation test.

		Parameters
		----------
		nrand : int
			Number of randomizations to perform.
		verbose : bool
			If True (default), displays a progress bar.

		Returns
		-------
		self : object
		"""
		self.compute_mosaic_residuals()
		self._compute_p_value(nrand=nrand, verbose=verbose)
		return self


	def summary(self, coordinate_index=None):
		"""
		Produces an output summarizing the test.

		Parameters
		----------
		coordinate_index : pd.Index or np.array
			Array of index information for the dimensions of the test statistic,
			when using a multidimensional test statistic. Otherwise ignored. 

		Returns
		-------
		summary : pd.DataFrame
			dataframe of summary information.
		"""
		fields = ['statistic', 'null_statistic_mean', 'p_value']
		d = self.null_statistics.shape[1]
		if d == 1:
			return pd.Series(
				[self.statistic, self.null_statistics.mean(), self.pval],
				index=fields
			)
		else:
			# Index
			if coordinate_index is None:
				coordinate_index = [f"Coordinate {i}" for i in range(d)]
			# Marginal p-values
			marg_pvals = 1 + np.sum(self.statistic <= self.null_statistics, axis=0).astype(float)
			marg_pvals /= float(1 + self.null_statistics.shape[0])
			# Construct dfs
			marg_out = pd.DataFrame(
				np.stack(
					[self.statistic, self.null_statistics.mean(axis=0), marg_pvals],
					axis=1
				),
				index=coordinate_index,
				columns=fields
			)
			adapt_out = pd.DataFrame(
				[[self.adapt_stat, self.null_adapt_stats.mean(), self.pval]],
				index=['Adaptive z-stat'],
				columns=fields,
			)
			out = pd.concat([adapt_out, marg_out], axis='index')
			out.index.name = 'Statistic type'
			return out

	def summary_plot(self, show_plot=False, **subplots_kwargs):
		"""
		Produces a plot summarizing the results of the test.

		Parameters
		----------
		show_plot : bool
			If true, shows the plot.
		**subplots_kwargs : dict
			kwargs for ``plt.subplots()``, e.g., ``figsize``.

		Returns
		-------
		fig : :class:`matplotlib.Figure`
			The figure from the ``plt.subplots()`` call.
		ax: array of :class:`matplotlib.Axes`
			Axes from the ``plt.subplots()`` call.
		"""
		import matplotlib.pyplot as plt
		import seaborn as sns
		fig, ax = plt.subplots(**subplots_kwargs)
		d = self.null_statistics.shape[1]
		if d == 1:
			nstats = self.null_statistics
			stat = self.statistic
		else:
			stat = self.adapt_stat
			nstats = self.null_adapt_stats
		sns.histplot(
			nstats, 
			label='Null statistics',
			color='cornflowerblue',
			alpha=0.5,
			ax=ax,
		)
		zstat = str(np.around(self.apprx_zstat, 2))
		if self.pval < 1e-3:
			pval = "{:e}".format(self.pval)
		else:
			pval = str(np.around(self.pval, 3))
		# formatting for p-value
		ax.axvline(
			stat, 
			color='red',
			label=f'Observed statistic\n(Apprx. Z={zstat}, pval={pval})',
		)
		plt.legend()
		if show_plot:
			plt.show()
		return fig, ax

	def _compute_apprx_zstat_tseries(self):
		"""
		Minor helper function used within _compute_p_value_tseries.
		"""
		### Compute z-statistics
		if self.stats_tseries.shape[1] == 1:
			ystat = self.stats_tseries[:, 0]
			ynulls = self.null_tseries[:, :, 0]
		else:
			ystat = self.adapt_stats_tseries
			ynulls = self.null_adapt_tseries
		self.apprx_zstat_tseries = (ystat - ynulls.mean(axis=1)) / (ynulls.std(axis=1))


	def _compute_p_value_tseries(
		self, nrand: int, verbose: bool, n_timepoints: int, window: Optional[int], 
	) -> None:
		"""
		Computes the underlying p-values in :meth:`fit_tseries`.

		Parameters
		----------
		nrand : int
			Number of randomizations to perform.
		verbose : bool
			If True (default), displays a progress bar. 
		n_timepoints : int
			Computes p-values at ``n_timepoints`` evenly spaced timepoints.
		window : int
			Window size.

		Returns
		-------
		None : NoneType
		"""
		n_obs = self.outcomes.shape[0]
		nvals = min(n_obs, n_timepoints)
		# starts/ends of windows to compute test statistic
		if window is None:
			self.ends = np.round(np.linspace(0, n_obs, nvals+1)[1:]).astype(int)
			self.ends = np.sort(np.unique(self.ends[self.ends > 0]))
			self.starts = np.zeros(len(self.ends)).astype(int)
		else:
			self.ends = np.round(np.linspace(window, n_obs, nvals)).astype(int)
			self.ends = np.sort(np.unique(self.ends[self.ends > 0]))
			self.starts = np.maximum(0, self.ends - window).astype(int)
		nvals = len(self.ends)
		# compute tseries statistic
		self.stats_tseries = np.stack(
			[
				self.test_stat(self.residuals[start:end], **self.tstat_kwargs) 
				for start, end in zip(self.starts, self.ends)
			], 
			axis=0
		).reshape(nvals, -1)
		d = self.stats_tseries.shape[1]
		# compute null statistics
		self.null_tseries = np.zeros((nvals, nrand, d))
		for r in utilities.vrange(nrand, verbose=verbose):
			self.permute_residuals()
			self.null_tseries[:, r] = np.stack(
				[self.test_stat(self._rtilde[start:end], **self.tstat_kwargs) 
				for start, end in zip(self.starts, self.ends)],
				axis=0
			).reshape(nvals, -1)
		# Compute p-values
		self.pval_tseries = np.zeros(nvals)
		self.adapt_stats_tseries = np.zeros(nvals)
		self.null_adapt_tseries = np.zeros((nvals, nrand))
		for i in range(nvals):
			out = compute_adaptive_pval(
				self.stats_tseries[i], self.null_tseries[i]
			)
			self.pval_tseries[i] = out[0]
			self.adapt_stats_tseries[i] = out[1]
			self.null_adapt_tseries[i] = out[2]

		### Compute z-statistics
		self._compute_apprx_zstat_tseries()

	def fit_tseries(
		self, 
		nrand: int=500,
		verbose: bool=True, 
		n_timepoints: int=20,
		window: Optional[int]=None, 
	):
		"""
		Runs mosaic permutation tests for various windows of the data,
		producing a time series of p-values.

		Parameters
		----------
		nrand : int
			Number of randomizations to perform.
		verbose : bool
			If True (default), displays a progress bar. 
		n_timepoints : int
			Computes p-values at ``n_timepoints`` evenly spaced
			timepoints. Default: 20.
		window : int
			Window size. Default: None (use all available data).

		Returns
		-------
		self : object
		"""
		self.compute_mosaic_residuals()
		self._compute_p_value_tseries(
			nrand=nrand, verbose=verbose, n_timepoints=n_timepoints, window=window
		)
		return self

	def plot_tseries(
		self, 
		time_index=None,
		alpha: float=0.05,
		show_plot: bool=True,
		**figure_kwargs
	) -> None:
		"""
		Plots the results of :meth:`fit_tseries`.

		Parameters
		----------
		time_index : np.array or pd.Index
			n_obs-length index information (e.g. datetimes) for each observation.
		alpha : float
			Nominal level.
		show_plot : bool	
			If True, runs ``matplotlib.pyplot.show()``.
		**figure_kwargs : dict
			kwargs for ``plt.figure()``, e.g., ``figsize``.

		Returns
		-------
		None : NoneType
			Returns None if show_plot=True. Else returns the following.
		fig : :class:`matplotlib.Figure`
			The figure from the ``plt.subplots()`` call.
		ax: array of :class:`matplotlib.Axes`
			Axes from the ``plt.subplots()`` call.
		"""
		# Create x-values
		if time_index is None:
			xvals = self.ends-1
		else:
			xvals = time_index[self.ends-1]

		# statistic value, null mean, and quantile
		if self.stats_tseries.shape[1] == 1:
			ystat = self.stats_tseries[:, 0]
			ynulls = self.null_tseries[:, :, 0]
		else:
			ystat = self.adapt_stats_tseries
			ynulls = self.null_adapt_tseries
		# Compute quantile 
		nrand = ynulls.shape[1]
		ynull_mean = ynulls.mean(axis=1)
		yquant = np.concatenate([ystat.reshape(-1, 1), ynulls], axis=1) # shape: nvals x (nrand + 1)
		yquant = np.sort(yquant, axis=1)
		rank = int(nrand + 1 - np.floor(alpha * (nrand + 1)))
		yquant = yquant[:, rank-1] # the -1 accounts for zero indexing
		return _create_tseries_plot(
			xvals=xvals,
			ystat=ystat,
			pvals=self.pval_tseries,
			zapprx=self.apprx_zstat_tseries,
			ynull_mean=ynull_mean,
			yquant=yquant,
			alpha=alpha,
			show_plot=show_plot,
			**figure_kwargs
		)

### Methods of aggregation
def _combine_pvals(pvals: np.array, how_combine: str='mean'):
	how_combine = str(how_combine).lower()
	if how_combine in ['mean', 'average']:
		return np.minimum(1, 2 * np.mean(pvals, axis=0))
	elif how_combine == 'median':
		return np.minimum(1, 2 * np.median(pvals, axis=0))
	elif how_combine == 'min':
		return np.minimum(1, len(pvals) * np.min(pvals, axis=0))
	else:
		raise ValueError(f"Unrecognized how_combine={how_combine}.")

def combine_mosaic_tests(
	mosaic_objects: list[MosaicPermutationTest],
	how_combine_pvals='median',
):
	"""
	Combines results across :class:`MosaicPermutationTest` objects.

	Parameters
	----------
	mosaic_objects : list
		List of objects inheriting from :class:`MosaicPermutationTest`,
		after calling ``.fit()`` for each object.
	how_combine_pvals : str
		How to combine the p-values. One of several options:

		- 'median' : twice the median
		- 'mean' : twice the mean
		- 'min' : len(mosaic_objects) times the minimum (Bonferroni correction).

	Returns
	-------
	summary : pd.Series
		Series of key inferential results.

	Notes
	-----
	This function combines results when using the same set 
	of test statistics and different tilings. To combine results
	using the same tiling and different test statistics,
	just use a vector-valued test statistic.
	"""
	mpts = mosaic_objects
	try:
		pval = _combine_pvals(
			pvals=np.array([mpt.pval for mpt in mpts]),
			how_combine=how_combine_pvals,
		)
		apprx_zstat = np.mean([mpt.apprx_zstat for mpt in mpts])
		# decide whether or not to use the adaptive variant
		d = mpts[0].null_statistics.shape[1]
		if d == 1:
			stat = np.mean([mpt.statistic for mpt in mpts])
			null_stat = np.mean([np.mean(mpt.null_statistics) for mpt in mpts])
		else:
			stat = np.mean([mpt.adapt_stat for mpt in mpts])
			null_stat = np.mean([np.mean(mpt.null_adapt_stats) for mpt in mpts])
		# Return summary
		return pd.Series(
			np.array([stat, null_stat, apprx_zstat, pval]),
			index=['statistic', 'null_statistic_mean', 'apprx_zstat', 'p-value'],
		)
	except AttributeError as e:
		raise AttributeError("combine_mosaic_tests raised AttributeError---try calling .fit()?")

def combine_mosaic_tests_tseries(
	mosaic_objects,
	how_combine_pvals='median',
	plot: bool=True,
	alpha: float=0.05,
	**figure_kwargs,
):
	"""
	Combines time-series results across :class:`MosaicPermutationTest` objects.

	Parameters
	----------
	mosaic_objects : list
		List of objects inheriting from :class:`MosaicPermutationTest`,
		after calling ``.fit_tseries()`` for each object.
	how_combine_pvals : str
		How to combine the p-values. One of several options:

		- 'median' : twice the median
		- 'mean' : twice the mean
		- 'min' : len(mosaic_objects) times the minimum (Bonferroni correction).
	plot : bool
		If True, displays a plot of results similar to the 
		``MosaicPermutationTest.plot_tseries()`` method.
	alpha : float
		Nominal level for error control.
	**figure_kwargs : dict
		kwargs for ``plt.figure()``, e.g., ``figsize``.

	Returns
	-------
	df : pd.DataFrame
		DataFrame of key inferential results.

	Notes
	-----
	This function combines results when using the same set 
	of test statistics and different tilings. To combine results
	using the same tiling and different test statistics,
	just use a vector-valued test statistic.
	"""
	mpts = mosaic_objects
	try:
		# Combine p-values and approximate z-statistics
		pvals = _combine_pvals(
			pvals=np.stack([mpt.pval_tseries for mpt in mpts], axis=0),
			how_combine=how_combine_pvals,
		)
		apprx_zstats = np.stack(
			[mpt.apprx_zstat_tseries for mpt in mpts], axis=0
		).mean(axis=0)
		# Combine statistics and null statistics
		if mpts[0].stats_tseries.shape[1] == 1:
			stat_tseries = np.stack(
				[mpt.stats_tseries.flatten() for mpt in mpts], axis=0
			).mean(axis=0)
			null_means = np.stack(
				[mpt.null_tseries[:, :, 0] for mpt in mpts], axis=0
			).mean(axis=0).mean(axis=1)
		else:
			stat_tseries = np.stack(
				[mpt.adapt_stats_tseries for mpt in mpts], axis=0
			).mean(axis=0)
			null_means = np.stack(
				[mpt.null_adapt_tseries for mpt in mpts], axis=0
			).mean(axis=0).mean(axis=1)
		# Dataframe to return
		df = pd.DataFrame(
			np.stack(
				[
					mpts[0].ends, stat_tseries, null_means, apprx_zstats, pvals
				],
				axis=1
			),
			columns=['Window end', 'statistic', 'null_statistic_mean', 'apprx_zstat', 'p-value'],
		)
		# Possibly plot
		if plot:
			show_plot = figure_kwargs.pop("show_plot", True)
			_create_tseries_plot(
				xvals=mpts[0].ends,
				ystat=stat_tseries,
				pvals=pvals,
				zapprx=apprx_zstats,
				ynull_mean=null_means,
				alpha=alpha,
				show_plot=show_plot,
				**figure_kwargs,
			)
		# Return
		return df
			
		
	except AttributeError as e:
		raise AttributeError(
			"combine_mosaic_tests_tseries raised AttributeError---try calling .fit_tseries()?"
		)
		
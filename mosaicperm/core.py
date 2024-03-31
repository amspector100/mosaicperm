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

		# compute p-value and return
		out = compute_adaptive_pval(
			self.statistic, self.null_statistics
		)
		self.pval, self.adapt_stat, self.null_adapt_stats = out
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
		**subplots_kwargs
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
		**subplots_kwargs : dict
			kwargs for ``plt.subplots()``, e.g., ``figsize``.

		Returns
		-------
		fig : :class:`matplotlib.Figure`
			The figure from the ``plt.subplots()`` call.
		ax: array of :class:`matplotlib.Axes`
			Axes from the ``plt.subplots()`` call.
		"""
		# Create plot and x-values
		import matplotlib.pyplot as plt
		subplots_kwargs['figsize'] = subplots_kwargs.get("figsize", (12, 6)) # default
		fig, axes = plt.subplots(1, 2, **subplots_kwargs)
		if time_index is None:
			xvals = self.ends-1
		else:
			xvals = time_index[self.ends-1]

		# Subplot 1: p-value
		zvals = np.maximum(stats.norm.ppf(1-self.pval_tseries), 0)
		axes[0].plot(xvals, zvals, color='blue', label='Observed')
		axes[0].scatter(xvals, zvals, color='blue')
		axes[0].axhline(
			stats.norm.ppf(1-alpha),
			color='black',
			linestyle='dotted',
			label=rf'Threshold ($\alpha$={alpha})'
		)
		axes[0].set(xlabel='Time', ylabel=r'Z-statistic: $\Phi(1-p)_+$')
		axes[0].legend()
		axes[0].set_ylim(0)
		# Subplot 2: statistic value and quantile
		if self.stats_tseries.shape[1] == 1:
			ystat = self.stats_tseries[:, 0]
			ynulls = self.null_tseries[:, :, 0]
		else:
			ystat = self.adapt_stats_tseries
			ynulls = self.null_adapt_tseries
		# Compute quantile 
		nrand = ynulls.shape[1]
		yquant = np.concatenate([ystat.reshape(-1, 1), ynulls], axis=1) # shape: nvals x (nrand + 1)
		yquant = np.sort(yquant, axis=1)
		rank = int(nrand + 1 - np.floor(alpha * (nrand + 1)))
		yquant = yquant[:, rank-1] # the -1 accounts for zero indexing
		# Plot
		axes[1].plot(xvals, ystat, color='cornflowerblue', label='Mosaic test statistic')
		axes[1].scatter(xvals, ystat, color='cornflowerblue')
		axes[1].plot(
			xvals, 
			yquant, 
			color='orangered',
			label=rf'Null quantile, $\alpha$={alpha}',
			linestyle='dotted',
		)
		axes[1].scatter(xvals, yquant, color='orangered')
		axes[1].set(xlabel='Time', ylabel='Statistic value')
		axes[1].legend()
		if show_plot:
			plt.show()
		else:
			return fig, axes
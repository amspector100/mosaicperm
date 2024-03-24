import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
from . import tilings, utilities, core
from typing import Optional


class MosaicFactorTest():

	def __init__(
		self,
		outcomes: np.array,
		exposures: np.array,
		test_stat: callable,
		test_stat_kwargs: Optional[dict]=None,
		tiles: Optional[list]=None, 
		**kwargs,
	):
		# Data
		self.outcomes = outcomes
		self.exposures = exposures
		# Test statistic
		self.test_stat = test_stat
		self.tstat_kwargs = test_stat_kwargs
		if self.tstat_kwargs is None:
			self.tstat_kwargs = {}
		# Tiles
		self.tiles = tiles
		if self.tiles is None:
			self.tiles = tilings.default_factor_tiles(
				exposures=self.exposures,
				n_obs=len(self.outcomes),
				**kwargs
			)

	def compute_mosaic_residuals(self):
		"""
		Computes mosaic-style residual estimates.
		
		Returns
		-------
		residuals : np.array
			(n_obs, n_subjects)-shaped array of residual estimates.
		"""
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

	def permute_residuals(self):
		"""
		Permutes residuals within tiles. Permuted values are stored in self._rtilde
		"""
		# Compute null variant
		for (batch, group) in self.tiles:
			# Permute within tile
			inds = np.arange(len(batch)); np.random.shuffle(inds)
			self._rtilde[np.ix_(batch, group)] = self.residuals[np.ix_(batch, group)][inds]


	def compute_p_value(self, nrand: int, verbose: bool):
		"""
		Parameters
		----------
		nrand : int
			Number of randomizations to perform.
		verbose : bool
			If True (default), displays a progress bar. 
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
		out = core.compute_adaptive_pval(
			self.statistic, self.null_statistics
		)
		self.pval, self.adapt_stat, self.null_adapt_stats = out
		return self.pval


	def fit(
		self,
		nrand: Optional[int]=500,
		verbose: Optional[bool]=True,
	):
		"""
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
		self.compute_p_value(nrand=nrand, verbose=verbose)
		return self


	def summary(self):
		fields = ['statistic', 'null_statistic_mean', 'p_value']
		d = self.null_statistics.shape[1]
		if d == 1:
			return pd.Series(
				[self.statistic, self.null_statistics.mean(), self.pval],
				index=fields
			)
		else:
			marg_pvals = 1 + np.sum(self.statistic <= self.null_statistics, axis=0).astype(float)
			marg_pvals /= float(1 + self.null_statistics.shape[0])
			marg_out = pd.DataFrame(
				np.stack(
					[self.statistic, self.null_statistics.mean(axis=0), marg_pvals],
					axis=1
				),
				index=[f"Coordinate {i}" for i in range(d)],
				columns=fields
			)
			adapt_out = pd.DataFrame(
				[[self.adapt_stat, self.null_adapt_stats.mean(), self.pval]],
				index=['Adaptive'],
				columns=fields,
			)
			out = pd.concat([adapt_out, marg_out], axis='index')
			out.index.name = 'Statistic type'
			return out

	def compute_p_value_tseries(
		self, nrand: int, verbose: bool, nvals: int, window: int, 
	):
		"""
		Parameters
		----------
		nrand : int
			Number of randomizations to perform.
		verbose : bool
			If True (default), displays a progress bar. 
		window : int
			Window size.
		nvals : int
			Number of evenly-spaced values to compute.
		"""
		n_obs = self.outcomes.shape[0]
		nvals = min(n_obs, nvals)
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
			out = core.compute_adaptive_pval(
				self.stats_tseries[i], self.null_tseries[i]
			)
			self.pval_tseries[i] = out[0]
			self.adapt_stats_tseries[i] = out[1]
			self.null_adapt_tseries[i] = out[2]

	def fit_tseries(
		self, 
		nrand: Optional[int]=500,
		verbose: Optional[bool]=True, 
		nvals: Optional[int]=20,
		window: Optional[int]=None, 
	):
		"""
		Parameters
		----------
		nrand : int
			Number of randomizations to perform.
		verbose : bool
			If True (default), displays a progress bar. 
		window : int
			Window size. Default: None.
		nvals : int
			Number of values. Defaults to 20.
		"""
		self.compute_mosaic_residuals()
		self.compute_p_value_tseries(
			nrand=nrand, verbose=verbose, nvals=nvals, window=window\
		)
		return self

	def plot_tseries(self, time_index=None, alpha=0.05, **subplots_kwargs):
		"""
		time_index : np.array or pd.Index
			n_obs-length index information (e.g. datetimes) for each observation.
		"""
		# Create plot and x-values
		import matplotlib.pyplot as plt
		subplots_kwargs['figsize'] = subplots_kwargs.get("figsize", (12, 6)) # default
		fig, axes = plt.subplots(1, 2, **subplots_kwargs)
		if time_index is None:
			xvals = self.ends
		else:
			xvals = time_index[self.ends]

		# Subplot 1: p-value
		zvals = np.maximum(stats.norm.ppf(1-self.pval_tseries), 0)
		axes[0].plot(xvals, zvals, color='blue', label='Observed')
		axes[0].scatter(xvals, zvals, color='blue')
		axes[0].axhline(
			stats.norm.ppf(1-alpha),
			color='black',
			linestyle='dotted',
			label=rf'Significance threshold\n(Marginal,  $\alpha$={alpha})'
		)
		axes[0].set(xlabel='', ylabel=r'Z-statistic: $\Phi(1-p)_+$')
		axes[0].legend()
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
		axes[1].set(xlabel='', ylabel='Statistic value')
		axes[1].legend()
		plt.show()
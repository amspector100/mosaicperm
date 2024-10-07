import time
import numpy as np
import pandas as pd
import unittest
import pytest
import os
import sys
import scipy.sparse
try:
	from . import context
	from .context import mosaicperm as mp
# For profiling
except ImportError:
	import context
	from context import mosaicperm as mp

class TestUtilities(context.MosaicTest):

	def test_get_sparse_dummies(self):
		np.random.seed(123)
		# Fake data
		n = 100; p = 5
		df = pd.DataFrame(
			np.random.binomial(4, 0.5, size=(n, p)),
			columns=[f'Feature {k}' for k in range(p)]
		)
		# expected output from pandas
		expected = pd.get_dummies(df, columns=df.columns, drop_first=False).values.astype(float)
		# our output
		output = mp.utilities.get_dummies_sparse(df, drop_first=False).todense()
		# test
		np.testing.assert_array_almost_equal(
			expected,
			output,
			decimal=8,
			err_msg=f"pd.get_dummies and get_dummies_sparse do not agree"
		)
		

if __name__ == "__main__":
	# Run tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestUtilities()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()
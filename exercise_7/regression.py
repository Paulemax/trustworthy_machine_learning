from typing import Tuple
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def regression_dataset(n:int=1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	w0 = 0.125
	b0 = 5.
	x_range = [-20, 60]

	def load_dataset(n:int=n) -> Tuple[np.ndarray, np.ndarray]:
		np.random.seed(43)

		def s(x:np.ndarray) -> np.ndarray:
			g = (x - x_range[0]) / (x_range[1] - x_range[0])
			return 3 * (0.25 + g**2.)

		x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
		eps = np.random.randn(n) * s(x)
		y = (w0 * x * (1. + np.sin(x)) + b0) + eps
		y = (y - y.mean()) / y.std()
		idx = np.argsort(x)
		x = x[idx]
		y = y[idx]
		return y, x

	Y, X = load_dataset()

	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,random_state=42)
	idxtrain = np.argsort(Xtrain)
	Xtrain=Xtrain[idxtrain]
	Ytrain=Ytrain[idxtrain]

	idxtest = np.argsort(Xtest)
	Xtest=Xtest[idxtest]
	Ytest=Ytest[idxtest]

	Xtrain = torch.tensor(Xtrain[:,None], dtype=torch.float)
	Ytrain = torch.tensor(Ytrain[:,None], dtype=torch.float)
	Xtest = torch.tensor(Xtest[:,None], dtype=torch.float)
	Ytest = torch.tensor(Ytest[:,None], dtype=torch.float)
	return Xtrain, Xtest, Ytrain, Ytest

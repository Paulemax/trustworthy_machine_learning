import sklearn.datasets
from sklearn.datasets._samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_moons(n:int=2000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float, float]]:
    
    Xin, yin = sklearn.datasets.make_moons(n_samples=2*n,noise=0.1)
    Xout, yout = sklearn.datasets.make_circles(n_samples=2*n,noise=0.2/3)
    Xout = Xout[yout==1]
    Xout = Xout*3
    Xout[:,0]=Xout[:,0]+0.5
    extent=[-3,3,-3,3]
    Xintrain, Xintest, yintrain, yintest = train_test_split(Xin,yin,stratify=yin,test_size=0.2)
    return Xintrain, yintrain, Xintest, yintest, Xout, extent



def create_blobs(n:int=2000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float, float]]:
    
    X, y = make_blobs(n_samples=4000, centers=3, cluster_std=0.60, random_state=0)
    Xin = X[y<2]
    yin = y[y<2]
    Xout = X[y>=2]
    extent=[-4,5,-1,6]
    Xintrain, Xintest, yintrain, yintest =train_test_split(Xin,yin,stratify=yin,test_size=0.2)
    
    return Xintrain, yintrain, Xintest, yintest, Xout, extent

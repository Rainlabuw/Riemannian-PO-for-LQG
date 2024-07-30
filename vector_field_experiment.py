"""In this script I want to figure out a faster way of computing orthgonal 
frames."""

from LQG_methods import *
import numpy as np
import control as ct
import matplotlib.pyplot as plt

n = 5
m = 3
p = 2
q = n
N = q*q + q*m + q*p
# init system
A = np.random.randn(n,n)
B = np.random.randn(n,m)
C = np.random.randn(p,n)
Q = np.eye(n)
R = np.eye(m)
V = np.random.randn(p,p)
V = V@V.T
W = np.random.randn(n,n)
W = W@W.T
w = (1,1,1)
sys = ControllerContainer(A, B, C, Q, R, W, V, q)
K = sys.rand()

def block2vec(K):
    K1, K2, K3 = sys.block2mat(K)
    K1 = np.reshape(K1, q*q)
    K2 = np.reshape(K2, q*p)
    K3 = np.reshape(K3, m*q)
    return np.concatenate([K1, K2, K3])

def vec2block(Kvec):
    K1 = Kvec[:q*q]
    K2 = Kvec[q*q:q*q + q*p]
    K3 = Kvec[q*q + q*p:]
    K1 = np.reshape(K1, (q,q))
    K2 = np.reshape(K2, (q,p))
    K3 = np.reshape(K3, (m,q))
    return sys.mat2block(K1, K2, K3)


W = np.eye(N)

def project(K, V, U):
    return sys.g(K, V, U)/sys.g(K, U, U)*U

def GS_process(K, basis):
    num_vectors = len(basis)
    v0 = basis[0]
    ON_basis = [v0/np.sqrt(sys.g(K, v0, v0, w))]
    for i in range(1, num_vectors):
        ON_basis


from LQG_methods import *
import numpy as np
import control as ct
import matplotlib.pyplot as plt

# initialize parameters 
np.random.seed(20)
num_steps = 300
alpha = .01
beta = .5
eps = 1e-6
s_bar = 1
n = 3
m = 2
p = 2

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
sys = ControllerContainer(A, B, C, Q, R, W, V)

# pick initial point
K0 = sys.rand()

# conduct Euclidean gradient descent
error_GD = sys.run_RGD_with_backtracking(
    num_steps, alpha, beta, eps, s_bar, K0
)

# conduct Riemannian gradient descent wrt KM-metric
error_RGD = sys.run_RGD_with_backtracking(
    num_steps, alpha, beta, eps, s_bar, K0, w=(1,1,1)
)

# plot
plt.figure()
plt.title(f"Random system with dim (n={n}, m={m}, p={p})")
plt.semilogy(error_GD, label="GD")
plt.semilogy(error_RGD, "--.", label="RGD")
plt.xlabel("iteration")
plt.ylabel(" (error) $J_n(K_t) - J_n^*$")
plt.grid()
plt.legend()
plt.show()
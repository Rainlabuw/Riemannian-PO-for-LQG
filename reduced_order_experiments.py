"""In this script, I'll be testing the reduced ordered experiments."""

from LQG_methods import *
import numpy as np
import control as ct
import matplotlib.pyplot as plt

#np.random.seed(2)

n = 4
m = 2
p = 2
q = 2

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

# pick initial point
K = sys.rand()
alpha = .01
beta = .5
eps = 1e-6
s_bar = 1
num_steps = 300
idx = 1
while True:
    cost = sys.LQG(K)
    K_dot = sys.natural_grad_LQG(K, w)
    K_dot_2norm = sys.g(K, K_dot, K_dot, w)
    s = s_bar
    Kplus = K - s*K_dot
    while not (sys.is_minimal(*sys.block2mat(Kplus)) and 
            sys.is_stabilizing(Kplus) and
            cost - sys.LQG(Kplus) >= alpha*s*K_dot_2norm):
        s *= .9
        Kplus = K - s*K_dot
        if s < 1e-100:
            raise ValueError("Backtracking failed.")
    K = Kplus
    
    print(
        f"idx: {idx}, \
        \t log(step size): {np.round(np.log10(s))}, \
        \t cost: {np.round(cost)}, \
        \t log(norm(grad)): {np.round(np.log10(K_dot_2norm))}"
    )
    if K_dot_2norm < 1e-5:
        print("Stationary point reached.")
        break
    idx += 1

H = np.zeros((sys.N, sys.N))
for i in range(sys.N):
    Ei = sys.basis[i]
    for j in range(sys.N):
        Ej = sys.basis[j]
        H[i,j] = sys.Hess(K, Ei, Ej)
eigs = sys.eig(H)
print(H.shape)
print(np.linalg.eig(H)[0])


# num_steps = 500
# alpha = .01
# beta = .5
# eps = 1e-6
# s_bar = 1

# error_GD = sys.run_RGD_with_backtracking(
#     10000, alpha, beta, eps, s_bar, K
# )

# error_RGD1 = sys.run_RGD_with_backtracking(
#     num_steps, alpha, beta, eps, s_bar, K, w=w
# )

# plt.figure()
# plt.semilogy(error_GD, label="GD")
# plt.semilogy(error_RGD1, label="RGD")
# plt.grid()
# plt.legend()
# plt.show()
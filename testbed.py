"""In this script, I just have some test im running."""


from LQG_methods import *
import numpy as np
import control as ct
import matplotlib.pyplot as plt

n = 2
m = 1
p = 1
I_n = np.eye(n)
I_m = np.eye(m)
I_p = np.eye(p)
#np.random.seed(2)

# init system
A12 = np.random.randn(n,n)
A21 = np.random.randn(n,n)
A = np.random.randn(n,n)
A_hat = np.block([
    [A, A12],
    [A21, A]
])

B = np.random.randn(n,m)
B_hat = np.block([
    [B, np.zeros((n,m))],
    [np.zeros((n,m)), B]
])
C = np.random.randn(p,n)
C_hat = np.block([
    [C, np.zeros((p,n))],
    [np.zeros((p,n)), C]
])
Q = np.eye(2*n)
R = np.eye(2*m)
V = np.eye(2*p)
W = np.eye(2*n)
w = (1,1,1)
sys_hat = ControllerContainer(A_hat, B_hat, C_hat, Q, R, W, V)
sys = ControllerContainer(A, B, C, I_n, np.eye(m), np.eye(n), np.eye(p))

def block2vec(K):
    K1, K2, K3 = sys.block2mat(K)
    K1 = np.reshape(K1, n*n)
    K2 = np.reshape(K2, n*p)
    K3 = np.reshape(K3, m*n)
    return np.concatenate([K1, K2, K3])

def vec2block(Kvec):
    K1 = Kvec[:n*n]
    K2 = Kvec[n*n:n*n + n*p]
    K3 = Kvec[n*n + n*p:]
    K1 = np.reshape(K1, (n,n))
    K2 = np.reshape(K2, (n,p))
    K3 = np.reshape(K3, (m,n))
    return sys.mat2block(K1, K2, K3)

def project(K, V, U):
    return sys.g(K, V, U, w)/sys.g(K, U, U, w)*U

def project_hat(K_hat, V_hat, U_hat):
    return sys_hat.g(
        K_hat, V_hat, U_hat, w
    )/sys_hat.g(K_hat, U_hat, U_hat, w)*U_hat

def normalize(K: np.ndarray, V: np.ndarray) -> np.ndarray:
    return V/np.sqrt(sys.g(K, V, V, w))

def normalize_hat(K_hat, V_hat):
    return V_hat/np.sqrt(sys_hat.g(K_hat, V_hat, V_hat, w))

basis = []
basis_matrix = []
for i in range(sys.N):
    partial_i = sys.basis[i]
    Ei, Fi, Gi = sys.block2mat(partial_i)
    Ej = np.zeros(Ei.shape)
    Fj = np.zeros(Fi.shape)
    Gj = np.zeros(Gi.shape)
    E_hat = np.block([
        [Ei, np.zeros((n,n))],
        [np.zeros((n,n)), Ej]
    ])
    F_hat = np.block([
        [Fi, np.zeros((n,p))],
        [np.zeros((n,p)), Fj]
    ])
    G_hat = np.block([
        [Gi, np.zeros((m,n))],
        [np.zeros((m,n)), Gj]
    ])
    partial_hat = sys_hat.mat2block(E_hat, F_hat, G_hat)
    basis.append(partial_hat)

for i in range(sys.N):
    partial_i = sys.basis[i]
    Ei, Fi, Gi = sys.block2mat(partial_i)
    Ej = np.zeros(Ei.shape)
    Fj = np.zeros(Fi.shape)
    Gj = np.zeros(Gi.shape)
    E_hat = np.block([
        [Ej, np.zeros((n,n))],
        [np.zeros((n,n)), Ei]
    ])
    F_hat = np.block([
        [Fj, np.zeros((n,p))],
        [np.zeros((n,p)), Fi]
    ])
    G_hat = np.block([
        [Gj, np.zeros((m,n))],
        [np.zeros((m,n)), Gi]
    ])
    partial_hat = sys_hat.mat2block(E_hat, F_hat, G_hat)
    basis.append(partial_hat)


r = 10
while True:
    A1_K = r*np.random.randn(n,n)
    A2_K = r*np.random.randn(n,n)
    B1_K = r*np.random.randn(n,p)
    B2_K = r*np.random.randn(n,p)
    C1_K = r*np.random.randn(m,n)
    C2_K = r*np.random.randn(m,n)
    A_K_hat = np.block([
        [A1_K, np.zeros((n,n))],
        [np.zeros((n,n)), A2_K]
    ])
    B_K_hat = np.block([
        [B1_K, np.zeros((n,p))],
        [np.zeros((n,p)), B2_K]
    ])
    C_K_hat = np.block([
        [C1_K, np.zeros((m,n))],
        [np.zeros((m,n)), C2_K]
    ])
    K_hat = sys_hat.mat2block(A_K_hat, B_K_hat, C_K_hat)
    if sys_hat.is_stabilizing(K_hat):
        break
K1 = sys.mat2block(A1_K, B1_K, C1_K)
K2 = sys.mat2block(A2_K, B2_K, C2_K)
xi_hat = sys_hat.natural_grad_LQG(K_hat, w)

basis_ON = [normalize_hat(K_hat, basis[0])]    
for i in range(1, len(basis)):
    Vi = basis[i]
    Ui = Vi
    for j in range(i):
        Uj = basis_ON[j]
        Ui = Ui - project_hat(K_hat, Vi, Uj)
    basis_ON.append(normalize_hat(K_hat, Ui))

xi_hat_proj = np.zeros(xi_hat.shape)
for E in basis_ON:
    xi_hat_proj += sys_hat.g(K_hat, E, xi_hat, w)*E
xi_hat = xi_hat_proj

H1 = np.random.randn(n,n)
H2 = np.random.randn(n,n)

V1 = sys.mat2block(H1@A1_K - A1_K@H1, H1@B1_K, -C1_K@H1)
V2 = sys.mat2block(H2@A2_K - A2_K@H2, H2@B2_K, -C2_K@H2)
E1, F1, G1 = sys.block2mat(V1)
E2, F2, G2 = sys.block2mat(V2)
E_hat = np.block([
    [E1, np.zeros((n,n))],
    [np.zeros((n,n)), E2]
])
F_hat = np.block([
    [F1, np.zeros((n,p))],
    [np.zeros((n,p)), F2]
])
G_hat = np.block([
    [G1, np.zeros((m,n))],
    [np.zeros((m,n)), G2]
])
V_hat = sys_hat.mat2block(E_hat, F_hat, G_hat)
print(sys_hat.g(K_hat, V_hat,  xi_hat, w))

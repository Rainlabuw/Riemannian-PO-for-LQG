"""In this script, Im trying to find a way to efficently compute the basis for
the horizontal space."""

from LQG_methods import *
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from time import time
from scipy.linalg import null_space
import cProfile

n = 5
m = 3
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
w = (1,1,1)
sys = ControllerContainer(A, B, C, Q, R, W, V)
N_H = n*m + n*p
N_V = n*n


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


def get_vertical_basis_mat(K):
    vertical_basis_mat = np.zeros((sys.N, N_V))
    A_K, B_K, C_K = sys.block2mat(K)
    k = 0
    for i in range(n):
        for j in range(n):
            H = np.zeros((n,n))
            H[i,j] = 1
            V = np.block([
                [np.zeros((m,p)), -C_K@H],
                [H@B_K, H@A_K - A_K@H]
            ])
            V_vec = block2vec(V)
            vertical_basis_mat[:,k] = V_vec
            k += 1
    return vertical_basis_mat

def get_vertical_basis(K):
    vertical_basis = []
    A_K, B_K, C_K = sys.block2mat(K)
    for i in range(n):
        for j in range(n):
            H = np.zeros((n,n))
            H[i,j] = 1
            V = np.block([
                [np.zeros((m,p)), -C_K@H],
                [H@B_K, H@A_K - A_K@H]
            ])
            vertical_basis.append(V)
    return vertical_basis

def get_horizontal_basis_mat(K):
    G = sys.g_coords(K, w)
    vertical_basis_mat = get_vertical_basis_mat(K)
    horizontal_basis_mat = null_space(vertical_basis_mat.T@G.T)
    return horizontal_basis_mat, G

def get_horizontal_basis(K):
    G = sys.g_coords(K, w)
    vertical_basis_mat = get_vertical_basis_mat(K)
    ker = null_space(vertical_basis_mat.T@G.T)
    horizontal_basis = []
    for i in range(N_H):
        H_vec = ker[:,i]
        H = vec2block(H_vec)
        horizontal_basis.append(H) 
    return horizontal_basis

def compute_grad_LQG(K):
    horizontal_basis_mat, G = get_horizontal_basis_mat(K)
    horizontal_G_coords = horizontal_basis_mat.T@G@horizontal_basis_mat
    b = np.zeros(N_H)
    for i in range(N_H):
        Hi = vec2block(horizontal_basis_mat[:,i])
        b[i] = sys.dLQG(K, Hi)
    x = np.zeros(N_H)
    L, low = cho_factor(horizontal_G_coords)
    x = cho_solve((L,low), b)
    V = np.zeros((n + m, n + p))
    for i in range(N_H):
        Hi = vec2block(horizontal_basis_mat[:,i])
        V += x[i]*Hi
    return V

def compute_grad_LQG_fast(K):
    vertical_basis_mat = np.zeros((sys.N, N_V))
    A_K, B_K, C_K = sys.block2mat(K)
    k = 0
    for i in range(n):
        for j in range(n):
            H = np.zeros((n,n))
            H[i,j] = 1
            V = np.block([
                [np.zeros((m,p)), -C_K@H],
                [H@B_K, H@A_K - A_K@H]
            ])
            V_vec = block2vec(V)
            vertical_basis_mat[:,k] = V_vec
            k += 1
    G = sys.g_coords(K, w)
    M = vertical_basis_mat.T@G.T
    _, _, v = np.linalg.svd(M, full_matrices=False)
    horizontal_basis_mat = v[N_V - N_H:,:].T
    print(sys.N, N_H, v.shape, horizontal_basis_mat.shape)
    horizontal_G_coords = horizontal_basis_mat.T@G@horizontal_basis_mat
    b = np.zeros(N_H)
    for i in range(N_H):
        Hi = vec2block(horizontal_basis_mat[:,i])
        b[i] = sys.dLQG(K, Hi)
    L, low = cho_factor(horizontal_G_coords)
    x = cho_solve((L,low), b)
    V = np.zeros((n + m, n + p))
    for i in range(n*m + n*p):
        Hi = vec2block(horizontal_basis_mat[:,i])
        V += x[i]*Hi
    return V

K = sys.rand()

def compute_grad_LQG_fast_2(K):
    vertical_basis_mat = get_vertical_basis_mat(K)
    _, _, v = np.linalg.svd(vertical_basis_mat.T)
    horizontal_basis_mat = v[-N_H:,:].T
    G = sys.g_coords(K, w)
    G_T_inv = np.linalg.inv(G.T)
    horizontal_basis_mat_2 = G_T_inv@horizontal_basis_mat
    horizontal_G_coords = horizontal_basis_mat.T@horizontal_basis_mat_2
    b = np.zeros(N_H)
    for i in range(N_H):
        Hi = vec2block(horizontal_basis_mat_2[:,i])
        b[i] = sys.dLQG(K, Hi)
    L, low = cho_factor(horizontal_G_coords)
    x = cho_solve((L,low), b)
    V = np.zeros((n + m, n + p))
    for i in range(n*m + n*p):
        Hi = vec2block(horizontal_basis_mat_2[:,i])
        V += x[i]*Hi
    return V


start_time = time()
V1 = compute_grad_LQG_fast_2(K)
print("basis: ", time() - start_time)

start_time = time()
V2 = compute_grad_LQG_fast(K)
print("reg: ", time() - start_time)


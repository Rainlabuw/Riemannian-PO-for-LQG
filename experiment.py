from LQG_methods import *
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import scipy as sp 


#seed = np.random.randint(0,1000)
#print(seed)

seed = 385
np.random.seed(seed)

def random_system(n, m, p, density=.8):
    A = sp.sparse.random(n, n, density=density).toarray()
    B = sp.sparse.random(n, m, density=density).toarray()
    C = sp.sparse.random(p, n, density=density).toarray()

    Q = np.eye(n)
    R = np.eye(m)
    W = np.eye(n)
    V = np.eye(p)
    return A, B, C, Q, R, W, V, n, m, p

def doyle_system():
    n = 2
    m = 1
    p = 1
    A = np.array([
        [1,1],
        [0,1]
    ])
    B = np.array([
        [0],
        [1]
    ])
    C = np.array([[1,0]])
    W = 5*np.array([
        [1,1],
        [1,1]
    ])
    V = np.array([[1]])
    Q = W.copy()
    R = V.copy()
    return A, B, C, Q, R, W, V, n, m, p

def nonminimal_system():
    n = 2
    m = 1
    p = 1
    A = np.array([
        [0,-1],
        [1,0]
    ])
    B = np.array([
        [1],
        [0]
    ])
    C = np.array([[1, -1]])
    W = np.array([
        [1,-1],
        [-1,16]
    ])
    V = np.array([[1]])
    Q = np.array([
        [4,0],
        [0,0]
    ])
    R = np.array([[1]])
    return A, B, C, Q, R, W, V, n, m, p

def zero_hessian_example_system():
    n = 2
    m = 1
    p = 1
    A = np.array([
        [-1,0],
        [1,-2]
    ])
    B = np.array([
        [-1],
        [1]
    ])
    C = np.array([[-2,11]])
    W = np.array([
        [1,0],
        [0,1]
    ])
    V = np.array([[1]])
    Q = W.copy()
    R = V.copy()
    return A, B, C, Q, R, W, V, n, m, p

def spring_mass_damper_system():
    n = 4
    m = 1
    p = 2
    Q = np.eye(n)
    R = np.eye(m)
    W = .01*np.eye(n)
    V = .01*np.eye(p)
    k1 = .5
    k2 = 1
    c1 = .2
    A = np.array([
        [0,0,1,0],
        [0,0,0,1],
        [-k1 - k2, k2, -c1, 0],
        [k2, -k2, 0, 0]
    ])
    B = np.array([
        [0],[0],[0],[1]
    ])
    C = np.array([
        [0,1,0,0],
        [0,0,0,1]
    ])
    return A, B, C, Q, R, W, V, n, m, p

def create_plot(
        system, 
        file_name, 
        system_name, 
        num_steps, 
        alpha, 
        beta, 
        eps, 
        s_bar, 
        subplot_index
    ):   
    A, B, C, Q, R, W, V, n, m, p = system()

    sys = ControllerContainer(A, B, C, Q, R, W, V)

    K0 = sys.rand()

    # r = 1e-8
    # while True:
    #     A_Kopt, B_Kopt, C_Kopt = sys.block2mat(sys.Kopt)
    #     A_K = A_Kopt + r*np.random.randn(n,n)
    #     B_K = B_Kopt + r*np.random.randn(n,p)
    #     C_K = C_Kopt + r*np.random.randn(m,n)
    #     K0 = sys.mat2block(A_K, B_K, C_K)
    #     if sys.is_stabilizing(K0) and sys.is_minimal(K0):
    #         break

    error_RGD2 = sys.run_RGD_with_backtracking(
        num_steps, alpha, beta, eps, s_bar, K0,w=(1,1,1)
    )
    error_RGD3 = sys.run_RGD_with_backtracking(
        num_steps, alpha, beta, eps, s_bar, K0, w=(1,0,0)
    )
    error_GD = sys.run_RGD_with_backtracking(
        num_steps, alpha, beta, eps, s_bar, K0
    )
    

    plt.subplot(2,2,subplot_index)
    #plt.title(system_name + f"\n(n={n}, m={m}, p={p})")
    plt.title(system_name)
    M = max(len(error_RGD2), len(error_RGD3))
    plt.semilogy(error_GD[:M], label="GD", linewidth=LINEWIDTH)
    plt.semilogy(error_RGD2, ":", color='orange',label="RGD w/ metric 1",linewidth=LINEWIDTH)
    plt.semilogy(error_RGD3, "--", color='green', label="RGD w/ metric 2",linewidth=LINEWIDTH)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if subplot_index in [1,3]:
        plt.ylabel("$J_n(K_t) - J_n^*$ (error)")
        plt.xlabel("iteration")
    plt.grid()

    if subplot_index == 1:
        plt.legend()


num_steps = 10000
alpha = .01
beta = .5
eps = 1e-6
s_bar = 1

LINEWIDTH = 3
FONTSIZE = 16

plt.figure(figsize=(8,10))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)



create_plot(
    lambda: random_system(4,3,3), 
    "random_system",
    "Random System", 
    num_steps, alpha, beta, eps, s_bar,4
)

create_plot(
    nonminimal_system, 
    "nonminimal_system",
    "System with Non-minimal LQG Controller", 
    num_steps, alpha, beta, eps, s_bar,1
)

create_plot(
    doyle_system, 
    "doyle_system",
    "Doyle System", 
    num_steps, alpha, beta, eps, s_bar,2
)

create_plot(
    zero_hessian_example_system, 
    "zero_hessian_system",
    "System with vanishing Hessian saddle point", 
    num_steps, alpha, beta, eps, s_bar,3
)

# create_plot(
#     spring_mass_damper_system, 
#     "spring_mass_system",
#     "Double Spring-Mass System", 
#     num_steps, alpha, beta, eps, s_bar,4
# )


plt.tight_layout()

#plt.savefig("./plot.pdf", format="pdf")
plt.show()
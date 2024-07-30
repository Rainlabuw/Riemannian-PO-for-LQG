# Created by Spencer Kraisler, 2024
# This script is an implementation of my paper Output-feedback Synthesis Orbit 
# Geometry: Quotient Manifolds and LQG Direct Policy Optimization by Kraisler and 
# Mesbahi.

import numpy as np
import control as ct
from scipy.linalg import sqrtm, cho_solve, cho_factor

# class LinearSystem:
#     def __init__(
#             self,
#             A: np.ndarray,
#             B: np.ndarray,
#             C: np.ndarray,
#             W: np.ndarray,
#             V: np.ndarray
#         ) -> None:
        
#         self.A = A # state matrix
#         self.B = B # input matrix
#         self.C = C # output matrix
#         self.W = W # system noise
#         self.V = V # measurement noise
#         self.n = A.shape[0] # state dim
#         self.m = B.shape[1] # input dim
#         self.p = C.shape[0] # output dim

#         # ensure (A,B,C) is in minimal state-space form
#         if not ControllerContainer.is_minimal(A, B, C):
#             raise ValueError("(A,B,C) is not minimal.")
        
#         # ensure (A,W) is controllable so LQG is well-defined
#         if not LinearSystem.is_controllable(A, sqrtm(W)):
#             raise ValueError("(A,sqrt(W)) is not controllable.")
        
#         # ensure (A,Q) is observable so LQG is well-defined
#         if not LinearSystem.is_observable(A, sqrtm(Q)):
#             raise ValueError("(A,sqrt(Q)) is not observable.")
        
#         self.I_n = np.eye(self.n)
#         self.I_m = np.eye(self.m)
#         self.I_p = np.eye(self.p)

#         self.basis = []
#         for i in range(self.n + self.m):
#             for j in range(self.n + self.p):
#                 if i >= self.m or j >= self.p:
#                     E = np.zeros((self.n + self.m, self.n + self.p))
#                     E[i,j] = 1
#                     self.basis.append(E)
#         self.N = len(self.basis)

#     @staticmethod
#     def eig(A: np.ndarray) -> np.ndarray:
#         """Returns eigenvalues of A. Made a wrapper so I do not have to index
#         the eig() method from NumPy."""
#         return np.linalg.eig(A)[0]

#     @staticmethod
#     def is_symmetric(A: np.ndarray, tol: float=1e-5) -> bool:
#         """Returns true if A is symmetric, within tolerance tol."""
#         return np.linalg.norm(A - A.T) < tol

#     @staticmethod
#     def is_positive_semidefinite(A: np.ndarray, tol: float = 1e-5) -> bool:
#         """returns true if A is positive semi-definite, within tolerance tol."""
#         return ControllerContainer.is_symmetric(A, tol) and \
#             min(ControllerContainer.eig(A)) >= 0
    
#     @staticmethod
#     def is_positive_definite(A: np.ndarray, tol: float = 1e-5) -> bool:
#         """returns true if A is positive definite, within tolerance tol."""
#         return ControllerContainer.is_symmetric(A, tol) and \
#             min(ControllerContainer.eig(A)) > tol

#     @staticmethod
#     def is_controllable(A: np.ndarray, B: np.ndarray) -> bool:
#         """Returns true if (A,B) is controllable and false otherwise."""
#         return np.linalg.matrix_rank(ct.ctrb(A, B)) == A.shape[0]
    
#     @staticmethod
#     def is_observable(A: np.ndarray, C: np.ndarray) -> bool:
#         """Returns true if (A,B) is observable and false otherwise."""
#         return np.linalg.matrix_rank(ct.obsv(A, C)) == A.shape[0]
    
#     @staticmethod
#     def is_minimal(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
#         """Returns true if (A,B) is controllable and (A,C) is observable, and
#         false otherwise."""
#         return ControllerContainer.is_controllable(A, B) and \
#             ControllerContainer.is_observable(A, C)

#     @staticmethod
#     def alpha(A: np.ndarray) -> float:
#         """Computes the spectral absicca of A. That is, max real part of the 
#         eigenvalues."""
#         return np.max(np.real(ControllerContainer.eig(A)))
    
#     @staticmethod
#     def is_stable(A: np.ndarray, tol:float = 1e-5) -> bool:
#         """returns true if A is Hurwitz stable, within tolerance tol."""
#         return ControllerContainer.alpha(A) < -tol

#     @staticmethod
#     def mat2block(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
#         """Constructs [0,C;B,A] block matrix from (A,B,C)."""
#         if A.shape[0] != B.shape[0]:
#             raise ValueError("(A,B) do not have compatible dimensions.")
#         if A.shape[0] != C.shape[1]:
#             raise ValueError("(A,C) do not have compatible dimensions.")
#         return np.block([
#             [np.zeros((C.shape[0], B.shape[1])), C],
#             [B, A]
#         ])
    
#     @staticmethod
#     def sym(Q: np.ndarray) -> np.ndarray:
#         """Computes the symmetric part of a matrix."""
#         return (Q + Q.T)/2
    
#     @staticmethod
#     def lyap(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
#         """Given Hurwitz stable A and positive semi-definite Q, computes the 
#         unique solution to AP + PA' = -Q. Note: This wrapper was created 
#         because sometimes ct.lyap(A,Q) returns an error if Q is just slightly
#         not symmetric. So, we assume Q is meant to be symmetric in this method.

#         Throws error if Q is not symmetric.
#         """
#         if not ControllerContainer.is_symmetric(Q):
#             raise ValueError("Q is not symmetric.")
#         return ct.lyap(A, ControllerContainer.sym(Q))
    
#     @staticmethod
#     def dlyap(
#         A: np.ndarray, Q: np.ndarray, E: np.ndarray, F: np.ndarray
#     ) -> np.ndarray:
#         """Returns the differential of the Lyapunov operator at (A,Q) along 
#         direction (E,F). Here, E,F are arbitrary matrices of the correct 
#         dimensions; A is Hurwitz stable, and Q is positive semi-definite."""
#         return ControllerContainer.lyap(
#             A, 
#             E@ControllerContainer.lyap(A, Q) + \
#                 ControllerContainer.lyap(A, Q)@E.T + F
#         )
    
#     def block2mat(self, P: np.ndarray) -> np.ndarray:
#         """Returns the system matrices A,B,C from the given block matix P."""
#         if P.shape != (self.n + self.m, self.n + self.p):
#             raise ValueError("P has incorrect dimensions.")
#         A = P[-self.n:, -self.n:]
#         B = P[-self.n:, :self.p]
#         C = P[:self.m, -self.n:]
#         return A, B, C

#     def coords_trans(self, T: np.ndarray, P: np.ndarray) -> np.ndarray:
#         """Performs a coordinate transformation on the internal state of the 
#         given system P via the similarity transformation T."""
#         if T.shape != (self.n, self.n):
#             raise ValueError("T has incorrect dimensions.")
#         A, B, C = self.block2mat(P)
#         inv_T = np.linalg.inv(T)
#         return self.mat2block(T@A@inv_T, T@B, C@inv_T)

class ControllerContainer:
    """A container of various methods for conducting gradient descent over
    the domain of stabilizing output-feedback controllers for a given
    plant P. 
    
    This class also contains methods for condcuting Riemannian gradient descent
    with respect to the Krishnaprasad-Martin metric found in Dynamic 
    Output-feedback Synthesis Orbit Geometry: Quotient Manifolds and LQG Direct 
    Policy Optimization by Kraisler and Mesbahi. 
    
    All methods (that are not static) whose arguments are linear systems 
    (controller or not) require the system to be in block form 
    K=[0, C_K; B_K, A_K].
    
    Many of these methods like X(), Y(), Wcl(), etc. are from the paper Analysis 
    of the Optimization Landscape of Linear Quadratic Gaussian (LQG) Control by 
    Zheng, Yang, and Na Li. I tried my best to be consisent with the notation in
    that paper."""
    def __init__(
            self, 
            A: np.ndarray, 
            B: np.ndarray, 
            C: np.ndarray, 
            Q: np.ndarray, 
            R: np.ndarray, 
            W: np.ndarray,
            V: np.ndarray,
            q: int = None,
        ) -> None:
        
        self.A = A # state matrix
        self.B = B # input matrix
        self.C = C # output matrix
        self.Q = Q # state reward
        self.R = R # input reward
        self.W = W # system noise
        self.V = V # measurement noise
        self.n = A.shape[0] # state dim
        self.m = B.shape[1] # input dim
        self.p = C.shape[0] # output dim

        if q is None or q == self.n:
            self.q = self.n
            self.full_ordered = True
        else:
            self.q = q
            self.full_ordered = False

        # ensure (A,B,C) is in minimal state-space form
        if not ControllerContainer.is_minimal(A, B, C):
            raise ValueError("(A,B,C) is not minimal.")
        
        # ensure (A,W) is controllable so LQG is well-defined
        if not ControllerContainer.is_controllable(A, sqrtm(W)):
            raise ValueError("(A,sqrt(W)) is not controllable.")
        
        # ensure (A,Q) is observable so LQG is well-defined
        if not ControllerContainer.is_observable(A, sqrtm(Q)):
            raise ValueError("(A,sqrt(Q)) is not observable.")
        
        self.I_n = np.eye(self.n)
        self.I_m = np.eye(self.m)
        self.I_p = np.eye(self.p)
        self.I_q = np.eye(self.q)

        self.basis = []
        for i in range(self.q):
            for j in range(self.q):
                E = np.zeros((self.q + self.m, self.q + self.p))
                E[self.m + i, self.p + j] = 1
                self.basis.append(E)
        for i in range(self.q):
            for j in range(self.p):
                E = np.zeros((self.q + self.m, self.q + self.p))
                E[self.m + i, j] = 1
                self.basis.append(E)
        for i in range(self.m):
            for j in range(self.q):
                E = np.zeros((self.q + self.m, self.q + self.p))
                E[i, self.p + j] = 1
                self.basis.append(E)
        self.N = len(self.basis)

        if self.full_ordered:
            Fopt = ControllerContainer.lqr(self.A, self.B, self.Q, self.R)
            Lopt = ControllerContainer.lqr(self.A.T, self.C.T, self.W, self.V).T
            A_Kopt = A + B@Fopt + Lopt@C
            B_Kopt = -Lopt
            C_Kopt = Fopt
            self.Kopt = self.mat2block(A_Kopt, B_Kopt, C_Kopt)
            self.LQG_opt = self.LQG(self.Kopt)

    @staticmethod
    def eig(A: np.ndarray) -> np.ndarray:
        """Returns eigenvalues of A. Made a wrapper so I do not have to index
        the eig() method from NumPy."""
        return np.linalg.eig(A)[0]

    @staticmethod
    def is_symmetric(A: np.ndarray, tol: float=1e-5) -> bool:
        """Returns true if A is symmetric, within tolerance tol."""
        return np.linalg.norm(A - A.T) < tol

    @staticmethod
    def is_positive_semidefinite(A: np.ndarray, tol: float = 1e-5) -> bool:
        """returns true if A is positive semi-definite, within tolerance tol."""
        return ControllerContainer.is_symmetric(A, tol) and \
            min(ControllerContainer.eig(A)) >= 0
    
    @staticmethod
    def is_positive_definite(A: np.ndarray, tol: float = 1e-5) -> bool:
        """returns true if A is positive definite, within tolerance tol."""
        return ControllerContainer.is_symmetric(A, tol) and \
            min(ControllerContainer.eig(A)) > tol

    @staticmethod
    def lqr(
        A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
    ) -> np.ndarray:
        """Returns continuous time LQR. This wrapper is just so that I have a 
        sign flip for the LQR (impling A + BK is stable), and so I do not have 
        to index the lqr() method from the python control library."""
        return -ct.lqr(A, B, Q, R)[0]

    @staticmethod
    def is_controllable(A: np.ndarray, B: np.ndarray) -> bool:
        """Returns true if (A,B) is controllable and false otherwise."""
        return np.linalg.matrix_rank(ct.ctrb(A, B)) == A.shape[0]
    
    @staticmethod
    def is_observable(A: np.ndarray, C: np.ndarray) -> bool:
        """Returns true if (A,B) is observable and false otherwise."""
        return np.linalg.matrix_rank(ct.obsv(A, C)) == A.shape[0]
    
    @staticmethod
    def is_minimal(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
        """Returns true if (A,B) is controllable and (A,C) is observable, and
        false otherwise."""
        return ControllerContainer.is_controllable(A, B) and \
            ControllerContainer.is_observable(A, C)

    @staticmethod
    def alpha(A: np.ndarray) -> float:
        """Computes the spectral absicca of A. That is, max real part of the 
        eigenvalues."""
        return np.max(np.real(ControllerContainer.eig(A)))
    
    @staticmethod
    def is_stable(A: np.ndarray, tol:float = 1e-5) -> bool:
        """returns true if A is Hurwitz stable, within tolerance tol."""
        return ControllerContainer.alpha(A) < -tol

    @staticmethod
    def mat2block(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Constructs [0,C;B,A] block matrix from (A,B,C)."""
        if A.shape[0] != B.shape[0]:
            raise ValueError("(A,B) do not have compatible dimensions.")
        if A.shape[0] != C.shape[1]:
            raise ValueError("(A,C) do not have compatible dimensions.")
        return np.block([
            [np.zeros((C.shape[0], B.shape[1])), C],
            [B, A]
        ])
    
    @staticmethod
    def sym(Q: np.ndarray) -> np.ndarray:
        """Computes the symmetric part of a matrix."""
        return (Q + Q.T)/2
    
    @staticmethod
    def lyap(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Given Hurwitz stable A and positive semi-definite Q, computes the 
        unique solution to AP + PA' = -Q. Note: This wrapper was created 
        because sometimes ct.lyap(A,Q) returns an error if Q is just slightly
        not symmetric. So, we assume Q is meant to be symmetric in this method.

        Throws error if Q is not symmetric.
        """
        if not ControllerContainer.is_symmetric(Q):
            raise ValueError("Q is not symmetric.")
        return ct.lyap(A, ControllerContainer.sym(Q))
    
    @staticmethod
    def dlyap(
        A: np.ndarray, Q: np.ndarray, E: np.ndarray, F: np.ndarray
    ) -> np.ndarray:
        """Returns the differential of the Lyapunov operator at (A,Q) along 
        direction (E,F). Here, E,F are arbitrary matrices of the correct 
        dimensions; A is Hurwitz stable, and Q is positive semi-definite."""
        return ControllerContainer.lyap(
            A, 
            E@ControllerContainer.lyap(A, Q) + \
                ControllerContainer.lyap(A, Q)@E.T + F
        )
    
    def rand_pole_placement(self) -> np.ndarray:
        """Returns a random stabilizing output feedback system. Generation is 
        done by constructing a random filter and random state-feedback 
        stabilizer with eigenvalues between -2 and -1."""

        F = -ct.place(self.A, self.B, np.random.rand(self.n) - 2)
        L = -ct.place(self.A.T, self.C.T, np.random.rand(self.n) - 2).T
        A_K = self.A + self.B@F + L@self.C
        B_K = -L
        C_K = F
        K = self.mat2block(A_K, B_K, C_K)
        return K
    
    def rand(self) -> np.ndarray:
        """Returns a random stabilizing output feedback system. K is generated 
        by choosing a random K r=.5 distance away from K0."""

        if self.full_ordered:
            r = .01
            while True:
                A_Kopt, B_Kopt, C_Kopt = self.block2mat(self.Kopt)
                A_K = A_Kopt + r*np.random.randn(self.n, self.n)
                B_K = B_Kopt + r*np.random.randn(self.n, self.p)
                C_K = C_Kopt + r*np.random.randn(self.m, self.n)
                K = self.mat2block(A_K, B_K, C_K)
                if self.is_stabilizing(K) and self.is_minimal(A_K, B_K, C_K):
                    break
        else:
            r = 10
            while True:
                A_K = r*np.random.randn(self.q, self.q)
                B_K = r*np.random.randn(self.q, self.p)
                C_K = r*np.random.randn(self.m, self.q)
                K = self.mat2block(A_K, B_K, C_K)
                if self.is_stabilizing(K) and self.is_minimal(A_K, B_K, C_K):
                    break
        return K
    
    def block2mat(self, P: np.ndarray) -> np.ndarray:
        """Returns the system matrices A,B,C from the given block matix P."""
        if P.shape != (self.q + self.m, self.q + self.p):
            raise ValueError("P has incorrect dimensions.")
        A = P[-self.q:, -self.q:]
        B = P[-self.q:, :self.p]
        C = P[:self.m, -self.q:]
        return A, B, C

    def coords_trans(self, T: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Performs a coordinate transformation on the internal state of the 
        given system P via the similarity transformation T."""
        if T.shape != (self.n, self.n):
            raise ValueError("T has incorrect dimensions.")
        A, B, C = self.block2mat(P)
        inv_T = np.linalg.inv(T)
        return self.mat2block(T@A@inv_T, T@B, C@inv_T)
    
    def is_stabilizing(self, K: np.ndarray) -> bool:
        """Checks if K is a stabilizing matrix."""
        return self.is_stable(self.Acl(K))

    def Acl(self, K: np.ndarray) -> np.ndarray:
        """Constructs the closed-loop state matrix with the controller K."""
        A_K, B_K, C_K = self.block2mat(K)
        return np.block([
            [self.A, self.B@C_K],
            [B_K@self.C, A_K]
        ])
    
    def Bcl(self, K: np.ndarray) -> np.ndarray:
        """Constructs the closed-loop input matrix with the controller K."""
        _, B_K, _ = self.block2mat(K)
        return np.block([
            [self.I_n, np.zeros((self.n, self.p))],
            [np.zeros((self.q, self.n)), B_K]
        ])

    def Ccl(self, K: np.ndarray) -> np.ndarray:
        """Constructs the closed-loop output matrix with the controller K."""
        _, _, C_K = self.block2mat(K)
        return np.block([
            [self.C, np.zeros((self.p, self.q))],
            [np.zeros((self.m, self.n)), C_K]
        ])
    
    def Dcl(self) -> np.ndarray:
        """Constructs the closed-loop feedforward matrix. Note this is 
        independent of K"""
        out = np.zeros((self.p + self.m, self.n + self.p))
        out[-self.p:,-self.p:] = self.I_p
        return out
    
    def dAcl(self, V: np.ndarray) -> np.ndarray:
        """Constructs the differential of Acl(.) along V. Note: dAcl is
        independent of K."""
        E, F, G = self.block2mat(V)
        return np.block([
            [np.zeros((self.n, self.n)), self.B@G],
            [F@self.C, E]
        ])
    
    def dBcl(self, V: np.ndarray) -> np.ndarray:
        _, F, _ = self.block2mat(V)
        return np.block([
            [np.zeros((self.n, self.n)), np.zeros((self.n, self.p))],
            [np.zeros((self.q, self.n)), F]
        ])
    
    def dCcl(self, V: np.ndarray) -> np.ndarray:
        _, _, G = self.block2mat(V)
        return np.block([
            [np.zeros((self.p, self.n)), np.zeros((self.p, self.q))],
            [np.zeros((self.m, self.n)), G]
        ])
    
    def Wcl(self, K: np.ndarray) -> np.ndarray:
        """Computes the Wcl(.) operator at K."""
        _, B_K, _ = self.block2mat(K)
        return np.block([
            [self.W, np.zeros((self.n, self.q))],
            [np.zeros((self.q, self.n)), B_K@self.V@B_K.T]
        ])
    
    def X(self, K: np.ndarray) -> np.ndarray:
        """Computes the X(.) operator."""
        return self.lyap(self.Acl(K), self.Wcl(K))
    
    def dWcl(self, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        _, B_K, _ = self.block2mat(K)
        _, F, _ = self.block2mat(V)
        return np.block([
            [np.zeros((self.n, self.n)), np.zeros((self.n, self.q))],
            [np.zeros((self.q, self.n)), F@self.V@B_K.T + B_K@self.V@F.T]
        ])
    
    def dX(self, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Computes the differential of X(.) at K along V."""
        out = self.dlyap(
            self.Acl(K), 
            self.Wcl(K), 
            self.dAcl(V),
            self.dWcl(K, V)
        )
        return out

    def Y(self, K: np.ndarray) -> np.ndarray:
        """Computes the Y(.) operator at K."""
        _, _, C_K = self.block2mat(K)
        return self.lyap(
            self.Acl(K).T, 
            np.block([
                [self.Q, np.zeros((self.n, self.q))],
                [np.zeros((self.q, self.n)), C_K.T@self.R@C_K]
            ])
        )

    def Qcl(self, K: np.ndarray) -> np.ndarray:
        """Computes the Qcl(.) operator at K."""
        _, _, C_K = self.block2mat(K)
        return np.block([
            [self.Q, np.zeros((self.n, self.q))],
            [np.zeros((self.q, self.n)), C_K.T@self.R@C_K]
        ])
    
    def dQcl(self, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        _, _, C_K = self.block2mat(K)
        _, _, G = self.block2mat(V)
        return np.block([
            [np.zeros((self.n, self.n)), np.zeros((self.n, self.q))],
            [np.zeros((self.q, self.n)), G.T@self.R@C_K + C_K.T@self.R@G]
        ])

    def LQG(self, K: np.ndarray) -> float:
        """Computes the LQG cost of the plant (A,B,C) with controller K."""
        
        return np.trace(self.Qcl(K)@self.X(K))
    
    def dLQG(self, K: np.ndarray, V:np.ndarray) -> float:
        """Computes the differential of the LQG cost at K along V."""

        return np.trace(
            self.dQcl(K, V)@self.X(K) + \
            self.Qcl(K)@self.dX(K, V)
        )
    
    def grad_LQG(self, K: np.ndarray) -> np.ndarray:
        """Computes the Euclidean gradient of the LQG cost at K."""

        _, B_K, C_K = self.block2mat(K)
        X = self.X(K)
        Y = self.Y(K)
        X11 = X[:self.n, :self.n]
        X12 = X[:self.n, self.n:]
        X22 = X[self.n:, self.n:]
        Y11 = Y[:self.n, :self.n]
        Y12 = Y[:self.n, self.n:]
        Y22 = Y[self.n:, self.n:]
        dA_K = 2*(Y12.T@X12 + Y22@X22)
        dB_K = 2*(Y22@B_K@self.V + Y22@X12.T@self.C.T + Y12.T@X11@self.C.T)
        dC_K = 2*(self.R@C_K@X22 + self.B.T@Y11@X12 + self.B.T@Y12@X22)
        return self.mat2block(dA_K, dB_K, dC_K)
    
    def g(
        self, K: np.ndarray, V: np.ndarray, W: np.ndarray, w: tuple, **kwargs
    ) -> float:
        """Computes the Krishnaprasad-Martin metric at K along vectors W and V.
        Here, w = (w1,w2,w3) are the parameters the metric. See the paper 
        Dynamic Output-feedback Synthesis Orbit Geometry: Quotient Manifolds and 
        LQG Direct Policy Optimization for more information."""

        if len(kwargs) == 0:
            sys = ct.StateSpace(
                self.Acl(K), self.Bcl(K), self.Ccl(K), self.Dcl()
            )
            Wc = ct.gram(sys, 'c')
            Wo = ct.gram(sys, 'o')
        else:
            Wc = kwargs['Wc']
            Wo = kwargs['Wo']

        E1_hat = self.dAcl(V)
        E2_hat = self.dAcl(W)
        F1_hat = self.dBcl(V)
        F2_hat = self.dBcl(W)
        G1_hat = self.dCcl(V)
        G2_hat = self.dCcl(W)

        return w[0]*np.trace(Wo@E1_hat@Wc@E2_hat.T) + \
            w[1]*np.trace(F1_hat.T@Wo@F2_hat) + \
            w[2]*np.trace(G1_hat@Wc@G2_hat.T)


    def g_coords(self, K: np.ndarray, w: tuple) -> np.ndarray:
        """Computes the coordinates of the KM metric at K (with parameters w).
        """
        sys = ct.StateSpace(self.Acl(K), self.Bcl(K), self.Ccl(K), self.Dcl())
        Wc = ct.gram(sys, 'c')
        Wo = ct.gram(sys, 'o')
        G = np.zeros((self.N, self.N))
        for i in range(self.N):
            Ei = self.basis[i]
            for j in range(i, self.N):
                Ej = self.basis[j]
                G[i,j] = self.g(K, Ei, Ej, w, Wc=Wc, Wo=Wo)
                G[j,i] = G[i,j]
        return G

    def natural_grad_LQG(
        self, K: np.ndarray, w: tuple
    ) -> np.ndarray:
        """Computes the Riemannian gradient of LQG at K wrt the KM metric 
        (with parameters w)."""
        G = self.g_coords(K, w)
        b = np.zeros(self.N)
        for i in range(self.N):
            Ei = self.basis[i]
            b[i] = self.dLQG(K, Ei)
        L, low = cho_factor(G)
        x = cho_solve((L,low), b) # Chomsky Decomp. to speed up performance.
        V = np.zeros((self.q + self.m, self.q + self.p))
        for i in range(self.N):
            Ei = self.basis[i]
            V += x[i]*Ei
        return V

    def run_RGD_with_backtracking(
        self, 
        num_steps: int, 
        alpha: float, 
        beta: float, 
        eps: float, 
        s_bar: float,
        K0: np.ndarray, 
        w: tuple=None
    ) -> np.ndarray:
        """Runs the Riemannian gradient descent algorthm with backtracking to
        determine the optimal step size. w is the parameters for the KM metric g."""

        error_hist = []
        K = K0.copy()
        for t in range(num_steps):
            LQG_K = self.LQG(K)
            if self.full_ordered:
                error = LQG_K - self.LQG_opt
            else:
                error = LQG_K
            if error < 1e-5:
                print("Machine precision reached.")
                break
            error_hist.append(error)
            if w == None:
                K_dot = self.grad_LQG(K)
                K_dot_2norm = np.linalg.norm(K_dot)**2
            else:
                K_dot = self.natural_grad_LQG(K, w)
                K_dot_2norm = self.g(K, K_dot, K_dot, w)
            if K_dot_2norm < eps**2:
                break
            s = s_bar
            Kplus = K - s*K_dot
            while not (
                self.is_stabilizing(Kplus) and 
                self.is_minimal(*self.block2mat(Kplus)) and
                LQG_K - self.LQG(Kplus) >= alpha*s*K_dot_2norm
            ):
                s *= beta 
                Kplus = K - s*K_dot
                if s < 1e-100:
                    raise ValueError("Backtracking failed.")
            K = Kplus 
            
            print(
                f"time: {t}, \
                LQG: {np.round(LQG_K, 3)}, \
                log(error): {np.round(np.log10(error), 3)}, \
                log(norm(Kdot)): {np.round(np.log10(K_dot_2norm), 3)}, \
                log(s): {np.round(np.log10(s), 3)}" \
            )
        return error_hist
    

    def sym_Hess(self, K, Delta):
        _, B_K, C_K = self.block2mat(K)
        Delta_A, Delta_B, Delta_C = self.block2mat(Delta)
        X_K = self.X(K)
        M1 = np.block([
            [np.zeros((self.n, self.n)), self.B@Delta_C],
            [Delta_B@self.C, Delta_A]
        ])@X_K + \
        X_K@np.block([
            [np.zeros((self.n, self.n)), self.B@Delta_C],
            [Delta_B@self.C, Delta_A]
        ]).T + \
        np.block([
            [np.zeros((self.n, self.n)), np.zeros((self.n, self.q))],
            [np.zeros((self.q, self.n)), B_K@self.V@Delta_B.T + Delta_B@self.V@B_K.T]
        ])
        X_K_prime = self.lyap(self.Acl(K), M1)
        Y_K = self.Y(K)
        A1 = np.block([
            [np.zeros((self.n, self.n)), self.B@Delta_C],
            [Delta_B@self.C, Delta_A]
        ])
        A2 = np.block([
            [np.zeros((self.n, self.n)), np.zeros((self.n, self.q))],
            [np.zeros((self.q, self.n)), C_K.T@self.R@Delta_C]
        ])
        A3 = np.block([
            [np.zeros((self.n, self.n)), np.zeros((self.n, self.q))],
            [np.zeros((self.q, self.n)), Delta_B@self.V@Delta_B.T]
        ])
        A4 = np.block([
            [np.zeros((self.n, self.n)), np.zeros((self.n, self.q))],
            [np.zeros((self.q, self.n)), Delta_C.T@self.R@Delta_C]
        ])
        return 2*np.trace(
            2*A1@X_K_prime@Y_K + 2*A2@X_K_prime + A3@Y_K + A4@X_K
        )
    
    def Hess(self, K, V1, V2):
        return 1/4*(self.sym_Hess(K, V1 + V2) - self.sym_Hess(K, V1 - V2))
# Riemannian-PO-for-LQG

This is an implementation of the policy optimization algorithm introduced in my paper **Dynamic Output-feedback Synthesis Orbit Geometry: Quotient Manifolds and LQG Direct Policy Optimization**.

This repo holds the methods for conducting gradient descent (GD) on the Linear-Quadratic-Gaussian (LQG) cost. It also performs Riemannian gradient descent (RGD) with respect to the Krishnaprasad-Martin (KM) metric introduced in the above paper. 

`LQG_methods.py` holds the methods container of everything needed to run policy optimization on your LQG problem setup. 

`main.py` runs a simple GD and RGD on a randomly generated system

`experiment.py` runs the exact experiment I included in the paper above.

Feel free to ask any questions on confusing parts, or raise issues for hidden bugs! :)

-Spencer Kraisler

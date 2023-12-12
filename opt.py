import numpy as np
from numpy.linalg import norm

def diag_hess_newtons_method(x0, f_grad_hess, args, max_iter=1000, tol=1e-6):
    x = x0.copy()
    num_iter = 0.0
    done = False
    while not done:
        f, grad, hess = f_grad_hess(x, **args)
        print(f)
        x -= grad / hess
        num_iter += 1
        done = (num_iter > max_iter) or norm(grad) < tol
    return x

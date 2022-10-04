import cvxpy as cp
import numpy as np
import scipy as sp
from numpy.linalg import inv, eig, pinv, norm


def l2_l2_hyper(y, Phi, lam, verbose=False):
    y = y.flatten()
    n, d = Phi.shape

    if n < d and lam == 0:
        return pinv(Phi)@y

    theta = cp.Variable(d)
    expr = 1/np.sqrt(n)*cp.norm(y-Phi@theta, 2) + lam*cp.norm(theta, 2)
    prob = cp.Problem(cp.Minimize(expr))
    prob.solve(verbose=verbose)
    return theta.value

def l2_l2(y, Phi, verbose=False):
    y = y.flatten()
    n, d = Phi.shape
    theta = cp.Variable(d)
    Sigma = Phi.T@Phi/n
    lam =  np.sqrt(np.trace(Sigma)/n)
    expr = 1/np.sqrt(n)*cp.norm(y - Phi@theta, 2) + lam*cp.norm(theta,2)
    prob = cp.Problem(cp.Minimize(expr))
    prob.solve(verbose=verbose)
    
    theta = theta.value
    W = Phi.T@Phi
    V = norm(y)*norm(y-Phi@theta)/np.sqrt(n)*np.eye(n)
    C = norm(y)*norm(theta)/np.sqrt(np.trace(W))*np.eye(d)
    return theta, C, V, lam

def l2_l1_hyper(y, Phi, lam, verbose=False):
    y = y.flatten()
    n, d = Phi.shape
    theta = cp.Variable(d)

    if n < d and lam == 0:
        return pinv(Phi)@y

    Sigma = Phi.T@Phi/n
    sqrtSigmad = np.diag(np.sqrt(np.diag(Sigma)))
    
    expr = 1/np.sqrt(n)*cp.norm(y - Phi@theta,2) +  lam*cp.norm(sqrtSigmad@theta,1)
    prob = cp.Problem(cp.Minimize(expr))
    prob.solve(verbose=verbose)
    
    theta = theta.value
    return theta

def l2_l1(y, Phi, verbose=False):
    y = y.flatten()
    n, d = Phi.shape
    theta = cp.Variable(d)
    Sigma = Phi.T@Phi/n
    sqrtSigmad = np.diag(np.sqrt(np.diag(Sigma)))
    lam = 1/np.sqrt(n) 
    expr = 1/np.sqrt(n)*cp.norm(y - Phi@theta,2) +  lam*cp.norm(sqrtSigmad@theta,1)
    prob = cp.Problem(cp.Minimize(expr))
    prob.solve(verbose=verbose)
    
    theta = theta.value
    V = norm(y)*norm(y-Phi@theta)/np.sqrt(n)*np.eye(n)
    W = np.diag(Phi.T@Phi)
    C = np.diag(norm(y)*(np.abs(theta)/np.sqrt(W)))
    return theta, C, V, lam

def l1_l1_hyper(y, Phi, lam, verbose=False):
    y = y.flatten()
    n, d = Phi.shape

    if n < d and lam == 0:
        return pinv(Phi)@y

    theta = cp.Variable(d)
    Sigma = Phi.T@Phi/n
    sqrtSigmad = np.diag(np.sqrt(np.diag(Sigma)))
    
    expr = 1/n*cp.norm(y - Phi@theta, 1) + lam*cp.norm(sqrtSigmad@theta,1)
    prob = cp.Problem(cp.Minimize(expr))
    prob.solve(verbose=verbose)
    
    theta = theta.value
    return theta

def l1_l1(y, Phi, verbose=False):
    y = y.flatten()
    n, d = Phi.shape
    theta = cp.Variable(d)
    Sigma = Phi.T@Phi/n
    sqrtSigmad = np.diag(np.sqrt(np.diag(Sigma)))
    lam = 1/np.sqrt(n) 
    expr = 1/n*cp.norm(y - Phi@theta, 1) + lam*cp.norm(sqrtSigmad@theta,1)
    prob = cp.Problem(cp.Minimize(expr))
    prob.solve(verbose=verbose)
    
    theta = theta.value
    V = np.diag(norm(y)*(np.abs(y-Phi@theta)))
    W = np.diag(Phi.T@Phi)
    C = np.diag(norm(y)*(np.abs(theta)/np.sqrt(W)))
    return theta, C, V, lam

def l1_l2_hyper(y, Phi, lam, verbose=False):
    y = y.flatten()
    n, d = Phi.shape

    if n < d and lam == 0:
        return pinv(Phi)@y

    theta = cp.Variable(d)

    expr = 1/n*cp.norm(y-Phi@theta, 1) + lam*cp.norm(theta, 2)
    prob = cp.Problem(cp.Minimize(expr))
    prob.solve(verbose=verbose)

    theta = theta.value
    return theta

def l1_l2(y, Phi, verbose=False):
    y = y.flatten()
    n, d = Phi.shape
    theta = cp.Variable(d)
    Sigma = Phi.T@Phi/n
    lam = np.sqrt( np.trace(Sigma)/n)
    expr = 1/n*cp.norm(y-Phi@theta, 1) + np.sqrt( np.trace(Sigma)/n)*cp.norm(theta, 2)
    prob = cp.Problem(cp.Minimize(expr))
    prob.solve(verbose=verbose)

    theta = theta.value
    W = Phi.T@Phi
    V = np.diag(norm(y)*(np.abs(y-Phi@theta)))
    C = norm(y)*norm(theta)/np.sqrt(np.trace(W))*np.eye(d)

    return theta, C, V, lam

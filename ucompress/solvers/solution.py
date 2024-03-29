import numpy as np

class Solution():
    """
    Class for storing the outputs of the solvers
    """
    def __init__(self, pars):

        N = pars.N
        Nt = pars.Nt

        self.t = pars.t

        self.u = np.zeros((N, Nt + 1))
        self.p = np.zeros((N, Nt + 1))
        self.lam_z = np.ones(Nt + 1)
        self.F = np.zeros(Nt + 1)
        self.J = np.zeros((N, Nt + 1))
        self.phi = pars.phi_0 * np.ones((N, Nt + 1))


    def trim_solution(self, n):
        """
        Trims the solution arrays.  Used when 
        Newton's method doesn't converge
        """

        self.u = self.u[:, :n]
        self.p = self.p[:, :n]
        self.lam_z = self.lam_z[:n]
        self.F = self.F[:n]
        self.J = self.J[:, :n]
        self.phi = self.phi[:, :n]
        
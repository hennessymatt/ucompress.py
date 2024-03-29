class Solution():
    """
    Class for storing the outputs of the solvers
    """
    def __init__(self, t, u, p, lam_z, F):

        self.t = t
        self.u = u
        self.p = p
        self.lam_z = lam_z
        self.F = F


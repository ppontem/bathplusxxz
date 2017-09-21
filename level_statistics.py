import numpy as np


class LevelStatistics:
    """
    Collection of methods to compute the level statistics of eigenvalues
    """
    def __init__(self, vals, t='f'):
        """
        Parameters
        ----------
        vals : eigenvalues
        t : type of eigenvalues: either 'f' (floquet) or 'h' (hamiltonian)
        """
        self.vals = np.copy(vals)
        self.type = t
        
    def get_r(self):
        """ Calculates the gap statistic r for every pair of neighbour eigenvalues
        """
        if self.type == 'f':
            phases = np.real(1.0j*np.log(self.vals))
            phases.sort()
            gaps = phases[1:] - phases[:-1]
            r = []
            for i in range(1, len(gaps)):
                r_i = min(gaps[i-1], gaps[i])/max(gaps[i-1], gaps[i])
                r.append(r_i)
            return np.array(r)
        elif self.type == 'h':
            vals_SORTED = np.sort(self.vals)
            gaps = vals_SORTED[1:] - vals_SORTED[:-1]
            r = []
            for i in range(1, len(gaps)):
                r_i = min(gaps[i-1], gaps[i])/max(gaps[i-1], gaps[i])
                r.append(r_i)
            return np.array(r)
        else:
            return "Type of the problem is neither floquet or hamiltonian"


    @staticmethod
    def goe_prediction(r):
        """ Probability distribution of r from the Gaussian Orthogonal Ensemble
        """
        return 2*27.0/8.0*(r + r**2)/(1 + r + r**2)**(2.5)


class LevelStatisticsPart:
    def __init__(self, vals, infT_frac=0.5, t='f'):
        """ Analyze the level statistics of only part of the spectrum closest
        to infinite temperature
        Parameters
        ----------
        vals : eigenvalues
        infT_frac : fraction of states around infinite temperature
        t : type of eigenvalues: either 'f' (floquet) or 'h' (hamiltonian)
        """
        inds_close_to_infT = np.argsort(np.abs(vals - np.mean(vals)))
        dd = len(vals)
        inds_close_to_infT_restricted = inds_close_to_infT[:int(dd*infT_frac)]
        self.vals = np.copy(vals[inds_close_to_infT_restricted])
        self.type = t
    
    def get_r(self):
        """ Calculates the gap statistic r for every pair of neighbour eigenvalues
        """
        if self.type == 'f':
            phases = np.real(1.0j*np.log(self.vals))
            phases.sort()
            gaps = phases[1:] - phases[:-1]
            r = []
            for i in range(1, len(gaps)):
                r_i = min(gaps[i-1], gaps[i])/max(gaps[i-1], gaps[i])
                r.append(r_i)
            return np.array(r)
        elif self.type == 'h':
            vals_SORTED = np.sort(self.vals)
            gaps = vals_SORTED[1:] - vals_SORTED[:-1]
            r = []
            for i in range(1, len(gaps)):
                r_i = min(gaps[i-1], gaps[i])/max(gaps[i-1], gaps[i])
                r.append(r_i)
            return np.array(r)
        else:
            return "Type of the problem is neither floquet or hamiltonian"


    @staticmethod
    def goe_prediction(r):
        """ Probability distribution of r from the Gaussian Orthogonal Ensemble
        """
        return 2*27.0/8.0*(r + r**2)/(1 + r + r**2)**(2.5)


#class LogRatiosOfGaps:
#    def __init__(self, vals, t='f'):
#        self.vals = np.copy(vals)
#        self.type = t
#    def get_log_ratios(self):
#        if self.type == 'f':
#            phases = np.real(1.0j*np.log(self.vals))
#            phases.sort()
#            gaps = phases[1:] - phases[:-1]
#            r = []
#            for i in range(1, len(gaps)):
#                r_i = min(gaps[i-1], gaps[i])/max(gaps[i-1], gaps[i])
#                r.append(r_i)
#            return np.array(r)
#        elif self.type == 'h':
#            vals_SORTED = np.sort(self.vals)
#            gaps = vals_SORTED[1:] - vals_SORTED[:-1]
#            r = []
#            gap_ratios = np.log(gaps[1:]/gaps[:-1])
#
#            return gap_ratios
#        else:
#            return "Type of the problem is neither floquet or hamiltonian"
#
#
#    @staticmethod
#    def goe_prediction(r):
#        return 2*27.0/8.0*(r + r**2)/(1 + r + r**2)**(2.5)

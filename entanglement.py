import numpy as np
import scipy.linalg
import collections


class EntanglementEntropy(object):
    """
    Collection of methods to compute the entanglement entropy of quantum states
    """
    def __init__(self, system_size, state):
        """
        Parameters
        ----------
        system_size : number of spins in the system
        state : eigenstate
        """
        self.system_size = system_size
        self.state = state

    def single_bit_entanglement(self, bit_labels):
        """ Calculates the entanglement entropy of each individual spin
        Parameters
        ----------
        bit_labels : labels of the bits for which we compute the entanglement 
                      with the rest of the system (from 1 to self.system_size)
        """
        ent_entropy_all_bits = collections.OrderedDict()
        for bit_label in bit_labels:
            if (bit_label == 1):
                shape_new = (2, 2**(self.system_size - 1))
                reshaped_state = np.reshape(self.state, shape_new)

            elif (bit_label == self.system_size):
                shape_new = (2**(self.system_size - 1), 2)
                reshaped_state = np.reshape(self.state, shape_new)

            else:
                shape_new = tuple([2**(bit_label-1)] + [2] + [2**(self.system_size - bit_label)])
                # print "shape_new", shape_new
                reshaped_state = np.reshape(self.state, shape_new)
                reshaped_state = np.transpose(reshaped_state, (1, 0, 2))
                reshaped_state = np.reshape(reshaped_state, (2, 2**(self.system_size - 1)))

            sing_vals = scipy.linalg.svdvals(reshaped_state)
            # print sing_vals
            probs = (sing_vals.real)**2
            # print "sum probs. :", np.sum(probs)
            # sing_vals[sing_vals < 0.0] = 1e-15
            ent_entropy_all_bits[bit_label] = 0.0
            # print
            for p in probs:
                if p > 0.0:
                    ent_entropy_all_bits[bit_label] += -p*np.log2(p)
        return ent_entropy_all_bits

    def subsystem_entanglement(self, bit_labels):
        """ Calculates the entanglement entropy of a subsystem
        Parameters
        ----------
        bit_labels : labels of the bits for which we compute the entanglement 
        with the rest of the system (from 1 to self.system_size)
        """
        bit_labels_inds = [i-1 for i in bit_labels]
        other_labels_inds = list(set(range(self.system_size)) - set(bit_labels_inds))
        state_reshape = np.reshape(self.state, tuple([2]*self.system_size))
        state_reshape = np.transpose(state_reshape, bit_labels_inds + other_labels_inds)
        state_reshape = np.reshape(state_reshape, (2**len(bit_labels_inds), 2**len(other_labels_inds)))
        sing_vals = scipy.linalg.svdvals(state_reshape)
        probs = (sing_vals.real)**2
        ent_entropy = 0.0
        # print
        for p in probs:
            if p > 0.0:
                ent_entropy += -p * np.log2(p)
        return ent_entropy

    def subsystem_entanglement_no_svd(self, bit_labels):
        """ Calculates the entanglement entropy of a subsystem (without SVD)
        Parameters
        ----------
        bit_labels : labels of the bits for which we compute the entanglement 
        with the rest of the system (from 1 to self.system_size)
        """
        rho = self.subsystem_reduced_density_matrix(bit_labels=bit_labels)
        vals = scipy.linalg.eigvalsh(rho)
        ent_entropy = 0.0
        # print
        for p in vals:
            if p > 0.0:
                ent_entropy += -p * np.log2(p)
        return ent_entropy

    def subsystem_reduced_density_matrix(self, bit_labels):
        """ Calculates the reduced density matrix of a subsystem 
        Parameters
        ----------
        bit_labels : labels of the bits for which we compute the reduced density matrix 
        (from 1 to self.system_size)
        """
        bit_labels_inds = [i-1 for i in bit_labels]
        other_labels_inds = list(set(range(self.system_size)) - set(bit_labels_inds))
        state_reshape = np.reshape(self.state, tuple([2]*self.system_size))
        state_reshape = np.transpose(state_reshape, bit_labels_inds + other_labels_inds)
        state_reshape = np.reshape(state_reshape, (2**len(bit_labels_inds), 2**len(other_labels_inds)))
        state_reshape = np.asmatrix(state_reshape)
        rho = state_reshape*state_reshape.H
        return rho


    def bipartite_entanglement(self, left_bits, right_bits):
        """ Calculates the bipartite entanglement entropy for 1d systems
        Parameters
        ----------
        left_bits : number of bits to the left of the cut (from 1
        to self.system_size)
        """
        ent_entropy_all_bits = collections.OrderedDict()
        # with kron last spins change first in the basis ordering, 
        # reshape fills along each row
        reshaped_state = np.reshape(self.state, (2**left_bits, 2**right_bits)) 
        sing_vals = scipy.linalg.svdvals(reshaped_state)
        # print sing_vals
        probs = sing_vals.real**2
        ent = 0.0
        for p in probs:
            if p > 0.0:
                ent += -p * np.log2(p)
        return ent

    def reduced_density_matrix(self, left_bits, right_bits, which='left'):
        """ Calculates the bipartite reduced density matrix for 1d systems
        Parameters
        ----------
        left_bits : number of bits to the left of the cut (from 1
        to self.system_size)
        which : DM for left or right part of the cut
        """
        reshaped_state = np.reshape(self.state, (2**left_bits,
                                                 2**right_bits))  
        reshaped_state = np.asmatrix(reshaped_state)
        if which == 'left':
            rho = reshaped_state*reshaped_state.H
        return rho

    @staticmethod
    def trace_distance(rho, sigma):
        """Calculate the trace distance between density matrices"""
        vals = np.linalg.eigvalsh(rho - sigma)
        return np.sum(np.abs(vals))

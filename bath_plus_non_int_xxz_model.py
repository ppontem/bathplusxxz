import scipy.linalg
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import LevelStatistics
import Entanglement
import scipy.spatial.distance


class BathPlusNonIntXXZModel(object):
    # Pauli matrices
    sigma_dict = {"x": scipy.sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]]),
                  "y": scipy.sparse.csr_matrix([[0.0, -1.0j], [1.0j, 0.0]]),
                  "z": scipy.sparse.csr_matrix([[1.0, 0.0], [0.0, -1.0]]),
                  "p": scipy.sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]]),
                  "m": scipy.sparse.csr_matrix([[0.0, 0.0], [1.0, 0.0]])}

    def __init__(self, N, M, h_list, J0, bath_seed, interactions_seed,
                 normalize_bath=1.0, couple=1):
        """Thermal bath coupled to non-integrable XXZ chain
	    Parameters
	    ----------
	    N : number of spins on the XXZ chain
	    M : number of spins in the bath
	    h_list : list of z-fields applied to chain spins (xxz part)
	    J0 : coupling of the bath to the XXZ chain
	    bath_seed : random seed for bath hamiltonian
	    interactions_seed : seed for the random matrices involved in the interactions
	    normalize_bath : constant that multiplies bath hamiltonian
        """
        self.N = N
        self.M = M

        if couple == 1:
            self.build_bath(bath_seed,
                            type='constant_width',
                            normalize=normalize_bath)
            self.J0 = J0
            self.xxz_chain_sz_sector_states()
            self.build_xxz_part(h_list)
            self.build_interaction_hamiltonian(J0, interactions_seed)
        else:
            self.J0 = J0
            self.xxz_chain_sz_sector_states()
            self.build_xxz_part(h_list)

        # self.sum_of_sz()

    @staticmethod
    def sigma(typ, i, L):
        """Generates pauli matrix of type typ in site i in the Hilbert space of
        L spins.
        Parameters
        ----------
        typ : pauli matrix type "x", "y", "z"
        i :  site for pauli matrix
        L : system size
        """
        ham = scipy.sparse.kron(BathPlusNonIntXXZModel.sigma_dict[typ],
                                scipy.sparse.identity(2**(L-i)))
        ham = scipy.sparse.kron(scipy.sparse.identity(2**(i-1)), ham)
        return ham

    @staticmethod
    def sample_GOE_random_matrix_unit_norm(M, seed):
        """Generates GOE random matrix with order 1 norm in the number of
        degrees of freedom.
        Parameters
        ----------
        M : Number of spins in the bath.
        seed : seed to reproduce GOE sample.
        """
        np.random.seed(seed)
        size = 2**M
        diagonal_elements = np.random.normal(loc=0.0,
                                             scale=np.sqrt(2.0/float(size)),
                                             size=size)
        off_diagonal_elements = [np.random.normal(loc=0.0,
                                                  scale=np.sqrt(1.0/float(size)),
                                                  size=diag_size) \ 
                                                  for diag_size in range(size-1,0,-1)]
        ham = scipy.sparse.diags(off_diagonal_elements,
                                 range(1, size),
                                 format='csr')
        ham = ham + ham.transpose()
        ham.setdiag(diagonal_elements)
        return ham

    @staticmethod
    def sample_goe_random_matrix_unit_norm_no_seed(M):
        """Generates GOE random matrix with order 1 norm in the number of
        degrees of freedom.
        Parameters
        ----------
        M : Number of spins in the bath
        """
        # np.random.seed(seed)
        size = 2**M
        diagonal_elements = np.random.normal(loc=0.0,
                                             scale=np.sqrt(2.0/float(size)),
                                             size=size)
        off_diagonal_elements = [np.random.normal(loc=0.0,
                                                  scale=np.sqrt(1.0/float(size)),
                                                  size=diag_size) \
												for diag_size in range(size-1,0,-1)]
        ham = scipy.sparse.diags(off_diagonal_elements, range(1, size), format='csr')
        ham = ham + ham.transpose()
        ham.setdiag(diagonal_elements)


        return ham

    def build_bath(self, seed, type="constant_width", normalize=1.0):
        """Builds the system bath: the width of the bath can be extensive in the
        number of degrees of freedom or have unit norm.
        Parameters
        ----------
        seed : seed for random matrix
        type : type of bath in terms of scaling of the width of eigenvalues, either
        constant_width or extensive_width
        """
        self.bath_type = type
        if self.bath_type == "constant_width":
            self.ham_bath_ = \
                normalize*self.sample_GOE_random_matrix_unit_norm(self.M, seed)

        elif self.bath_type == "extensive_width":
            self.ham_bath = self.sample_GOE_random_matrix_M_norm(self.M, seed)

    def build_xxz_part(self, h_list):
        """Builds the non-integrable XXZ chain
        Parameters
        ----------
        h_list : list of z-fields applied to the spins of the xxz chain
        """
        h_xxz = scipy.sparse.csr_matrix((2**(self.N), 2**(self.N)))
        for i in range(1, self.N):
            heis_xx = 0.25*self.sigma("x", i, self.N).dot(self.sigma("x", i+1, self.N))
            heis_yy = 0.25*self.sigma("y", i, self.N).dot(self.sigma("y", i+1, self.N))
            heis_zz = 0.25*self.sigma("z", i, self.N).dot(self.sigma("z", i+1, self.N))
            heis_z = 0.5*self.sigma("z", i, self.N)*h_list[i-1]
            h_xxz = h_xxz + heis_xx + heis_yy + heis_zz + heis_z
        h_xxz = h_xxz + 0.5*self.sigma("z", self.N, self.N)*h_list[-1]
        # add integrabilty breaking terms
        for i in range(1, self.N-1):
            heis_xx_2 = 0.25 * self.sigma("x", i, self.N).dot(self.sigma("x", i + 2, self.N))
            heis_yy_2 = 0.25 * self.sigma("y", i, self.N).dot(self.sigma("y", i + 2, self.N))
            h_xxz = h_xxz + heis_xx_2 + heis_yy_2
        self.h_xxz_ = h_xxz[self.good_numbs, :][:, self.good_numbs]
        return h_xxz

    def build_interaction_hamiltonian(self, J0, seed):
        """Builds the interaction part of the Hamiltonian between the bath and 
        the xxz chain.
        Parameters
        ----------
        J0 : coupling of the bath to the XXZ chain
        seed : initializes the random generator which builds the different random matrices
        """
        # interaction Hamiltonian
        np.random.seed(seed + 1234567)
        h_int = scipy.sparse.csr_matrix((2**(self.M)*len(self.good_numbs), 2**(self.M)*len(self.good_numbs)))
        B = self.sample_goe_random_matrix_unit_norm_no_seed(self.M)
        C = self.sample_goe_random_matrix_unit_norm_no_seed(self.M)
        D = self.sample_goe_random_matrix_unit_norm_no_seed(self.M)
        E = self.sample_goe_random_matrix_unit_norm_no_seed(self.M)
        h_int = h_int + \
            scipy.sparse.kron(B, self.sigma("z", 1, self.N).tocsr()[self.good_numbs, :][:, self.good_numbs])
        h_int = h_int + \
            scipy.sparse.kron(C, self.sigma("z", 1, self.N).dot(self.sigma("z", 2, self.N)).tocsr()[self.good_numbs, :][:, self.good_numbs])
        hop_12 = (self.sigma("m", 1, self.N).dot(self.sigma("p", 2, self.N)) +
                  self.sigma("p", 1, self.N).dot(self.sigma("m", 2, self.N))).tocsr()[self.good_numbs, :][:, self.good_numbs]
        h_int = h_int + scipy.sparse.kron(D, hop_12)
        # h_int = h_int + scipy.sparse.kron(D, )
        h_int = h_int + scipy.sparse.kron(E, self.sigma("z", 2, self.N).tocsr()[self.good_numbs, :][:, self.good_numbs])
        h_int = J0*h_int

        h_bath_full = scipy.sparse.kron(self.ham_bath_, scipy.sparse.identity(len(self.good_numbs)))
        h_xxz_full = scipy.sparse.kron(scipy.sparse.identity(2 ** self.M), self.h_xxz_)
        self.ham_ = h_bath_full + h_xxz_full + h_int

    def diagonalize(self, save_eig_vecs=True, lanczos=False, k=100):
        """Diagonalizes the Hamiltonian
        Parameters
        ----------
        save_eig_vecs : option to save the eig_vecs to self.eig_vecs_
        lanczos : option to use lanczos solver
        k : number of eigenstates if using lanczos
        """
        if not lanczos:
            self.eig_vals_, eig_vecs = scipy.linalg.eigh(self.ham_.todense())
        else:
            self.eig_vals_, eig_vecs = scipy.sparse.linalg.eigsh(self.ham_,
                                                                 k=k,
                                                                 sigma=self.ham_.diagonal().mean())
        if save_eig_vecs:
            self.eig_vecs_ = eig_vecs

    def diagonalize_bath(self):
        """Diagonalizes the bath
        """
        self.eig_vals_bath_, eig_vecs_bath = scipy.linalg.eigh(self.ham_bath_.todense())
        return eig_vecs_bath

    def diagonalize_xxz_part(self, lanczos=False, k=100):
        """Diagonalizes the XXZ part of the Hamiltonian
        Parameters
        ----------
        lanczos : option to use lanczos solver
        k : number of eigenstates if using lanczos
        """
        if not lanczos:
            self.eig_vals_xxz_, eig_vecs_xxz_ = scipy.linalg.eigh(self.h_xxz_.todense())
        else:
            self.eig_vals_xxz_, eig_vecs_xxz_ = \
                scipy.sparse.linalg.eigsh(self.h_xxz_,
                                          k=k,
                                          sigma=self.h_xxz_.diagonal().mean())

        return eig_vecs_xxz_

    def xxz_chain_sz_sector_states(self):
        """Get the states in the total spin zero sector of the hilbert space
        """
        self.good_numbs = []
        for i in range(2 ** self.N):
            sum_ = self.N - np.sum([int(j) for j in "{0:b}".format(i)])
            if sum_ == self.N/2:
                self.good_numbs.append(i)
        self.good_numbs = np.asarray(self.good_numbs)

    def sum_of_sz(self):
        """Total sigma^z operator
        """
        op_ = scipy.sparse.csr_matrix((2 ** (self.N), 2 ** (self.N)))
        for i in range(1, self.N+1):
            op_ = op_ + self.sigma("z", i, self.N)
        print op_.diagonal()[self.good_numbs]

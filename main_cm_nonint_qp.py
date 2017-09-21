import os
import sys
import inspect
import argparse
import bath_plus_non_int_xxz_model
import level_statistics
import entanglement
import numpy as np
import scipy.linalg
import time


def transform_to_full_hs(v, full_dim, good_numbs_full):
    """Transforms eigenstate in reduced Hilbert Space to larger one.
    Parameters
    ----------
    v : eigenstate
    full_dim : dimension of full hulbert space
    good_numbs_full : corresping states in the large hilbert space
    """
    vv = np.zeros(full_dim, dtype=complex)
    vv[good_numbs_full] = np.copy(v)
    return vv


parser = argparse.ArgumentParser(description='XXZ chain command line')
parser.add_argument('N', type=int, help='number of spins in xxz chain')
parser.add_argument('M', type=int, help='dofs in bath')
parser.add_argument('W', type=float, help='xxz fields drawn from uniform distribution [-W, W]')
parser.add_argument('J0', type=float, help='coupling strength')
parser.add_argument('samples', type=str, help='e.g. 1-100')
# parser.add_argument('random_seed', type=str, help='e.g. 1-150')
parser.add_argument('PERC_INF_T_EIGS', type=int, help='percentage of the total spectrum taken from around the infinite temperature energy value')
parser.add_argument('only_xxz', type=int, help='ignores the inclusion, only computes XXZ chain properties')
parser.add_argument('normalize_bath', type=float, help='multiplies bath random matrix')
parser.add_argument('MAIN_SAVE_FOLDER', type=str, help='e.g. main save folder')
args = parser.parse_args()

# Define the parameters of the model
N = args.N
M = args.M
J0 = args.J0
W = args.W
PEIGS = args.PERC_INF_T_EIGS
samples = args.samples
samples_split = samples.split('-')
samples_int = [int(i) for i in samples_split]
normalize_bath = args.normalize_bath
MAIN_SAVE_FOLDER = args.MAIN_SAVE_FOLDER

# Collect data about single l-bit entropy (S1), bipartite cuts entropy (SA),
# and level spacing statistics
S1_vs_lbit_vs_state_sample = []
SA_vs_cut_vs_state_sample = []
sample_id_vs_state_sample = []
stats_r_vs_state_sample = []
stats_r_sample_vs_samples = []

start = time.time()
for seed_ in range(samples_int[0], samples_int[1]+1):
    if seed_ % 50 == 0:
        print seed_
    np.random.seed(49 + seed_)
    h_list = W*np.cos(2*np.pi*(np.sqrt(5)-1)/2.0*np.arange(1.0, N+0.1, 1.0) \
                      + np.random.uniform(-np.pi, np.pi))
    if args.only_xxz != 1:
        model = bath_plus_non_int_xxz_model.BathPlusNonIntXXZModel(N, 
                                                                   M, 
                                                                   h_list, 
                                                                   J0, 
                                                                   seed_, 
                                                                   seed_,
                                                                   normalize_bath=normalize_bath)
        # eig_vecs_xxz_ = model.diagonalize_xxz_part()
        model.diagonalize()

    else:
        model = bath_plus_non_int_xxz_model.BathPlusNonIntXXZModel(N, 
                                                                   M, 
                                                                   h_list, 
                                                                   J0, 
                                                                   seed_, 
                                                                   seed_,
                                                                   normalize_bath=normalize_bath,
                                                                   couple=0)
        eig_vecs_xxz_ = model.diagonalize_xxz_part()
        model.eig_vecs_ = eig_vecs_xxz_
        model.eig_vals_ = model.eig_vals_xxz_
        M = 0




    full_dim = 2**N*2**M
    good_numbs_full = np.asarray([(model.good_numbs+ i_ * 2**N).tolist() for i_ in range(0, 2**M)]).flatten()

    #for i in range(len(model.eig_vals_)):
    NEIGS = int(PEIGS/100.0*len(model.eig_vals_))
    # Select states closest to infinite temperature
    indices_of_NEIGS_closest_to_inf_T = np.sort(
            np.argsort(
                    np.abs(
                            model.eig_vals_ - np.mean(model.eig_vals_)))[:NEIGS])
    # Calculate entanglement
    for i in indices_of_NEIGS_closest_to_inf_T:
        S1_vs_lbit = []
        SA_vs_cut = []
        eig_vec_ = model.eig_vecs_[:, i]
        vv_ = transform_to_full_hs(eig_vec_, full_dim, good_numbs_full)
        for lbit_ind in range(1,  M+N+1):
            S1_vs_lbit.append(Entanglement.EntanglementEntropy(N + M, vv_).subsystem_entanglement_no_svd([lbit_ind]))
            if lbit_ind < M+N:
                if lbit_ind <= (N+M)/2:
                    SA_ = Entanglement.EntanglementEntropy(N + M, vv_).subsystem_entanglement_no_svd(range(1, lbit_ind+1))
                else:
                    SA_ = Entanglement.EntanglementEntropy(N + M, vv_).subsystem_entanglement_no_svd(range(lbit_ind+1, M+N+1))
                SA_vs_cut.append(SA_)
        S1_vs_lbit_vs_state_sample.append(S1_vs_lbit)
        SA_vs_cut_vs_state_sample.append(SA_vs_cut)
    sample_id_vs_state_sample.extend([seed_]*len(indices_of_NEIGS_closest_to_inf_T))
    # Calculate level statistics
    stats_r_obj = level_statistics.LevelStatistics(model.eig_vals_[indices_of_NEIGS_closest_to_inf_T], t='h')
    stats_r = stats_r_obj.get_r()
    stats_r_vs_state_sample.extend(stats_r.tolist())
    stats_r_sample_vs_samples.extend([seed_]*len(stats_r))

# Save the results to disk
if not os.path.exists(MAIN_SAVE_FOLDER+"S_1/"):
    os.makedirs(MAIN_SAVE_FOLDER+"S_1/")
np.savez_compressed(MAIN_SAVE_FOLDER+"S_1/M_{}_J0_{:.8f}_N_{}_W_{:.5f}_norm_bath_{:.2f}_PINFEIGS_{}_onlyxxz_{}_samples_{}_{}".format(M, J0, N,
                                                                              W,
                                                                              normalize_bath,
                                                                              PEIGS,
                                                                              args.only_xxz,
                                                                              samples_int[0],
                                                                              samples_int[1]),
                    S_1_data=np.asarray(S1_vs_lbit_vs_state_sample),
                    S_1_sample_info=np.asarray(sample_id_vs_state_sample)[:, np.newaxis])
if not os.path.exists(MAIN_SAVE_FOLDER+"S_all_bipartitions/"):
    os.makedirs(MAIN_SAVE_FOLDER+"S_all_bipartitions/")
np.savez_compressed(MAIN_SAVE_FOLDER+"S_all_bipartitions/M_{}_J0_{:.8f}_N_{}_W_{:.5f}_norm_bath_{:.2f}_PINFEIGS_{}_onlyxxz_{}_samples_{}_{}".format(M, J0, N,
                                                                              W,
                                                                              normalize_bath,
                                                                              PEIGS,
                                                                              args.only_xxz,
                                                                              samples_int[0],
                                                                              samples_int[1]),
                    S_bipart_data=np.asarray(SA_vs_cut_vs_state_sample),
                    S_bipart_sample_info=np.asarray(sample_id_vs_state_sample)[:, np.newaxis])

if not os.path.exists(MAIN_SAVE_FOLDER+"spectral_data/"):
    os.makedirs(MAIN_SAVE_FOLDER+"spectral_data/")
np.savez_compressed(MAIN_SAVE_FOLDER+"spectral_data/M_{}_J0_{:.5f}_N_{}_W_{:.8f}_norm_bath_{:.2f}_PINFEIGS_{}_onlyxxz_{}_samples_{}_{}".format(M, J0, N,
                                                                              W,
                                                                              normalize_bath,
                                                                              PEIGS,
                                                                              args.only_xxz,
                                                                              samples_int[0],
                                                                              samples_int[1]),
                    **{
                       "stats_r": np.asarray(stats_r_vs_state_sample), "stats_r_sample_id": stats_r_sample_vs_samples
                       })

print "Total time: ", time.time() - start

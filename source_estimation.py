"""This file contains some functions needed to estimate (via maximum
likelihood) the source of a SI epidemic process (with Gaussian edge delays).

The important function is
    s_est, likelihood = ml_estimate(graph, obs_time, sigma, is_tree, paths,
    path_lengths, max_dist)

where s_est is the list of nodes having maximum a posteriori likelihood and
likelihood is a dictionary containing the a posteriori likelihood of every
node.

"""
import math
import networkx as nx
import numpy as np
import GLAD_MODIFIED.source_est_tools as tl
import operator
import collections

import scipy.stats as st
from scipy.misc import logsumexp

def ml_estimate(graph, obs_time, sigma, mu, paths, path_lengths,
        max_dist=np.inf):
    """Returns estimated source from graph and partial observation of the
    process.

    - graph is a networkx graph
    - obs_time is a dictionary containing the observervations: observer -->
      time

    Output:
    - list of nodes having maximum a posteriori likelihood
    - dictionary: node -> a posteriori likelihood

    """
    ### Gets the sorted observers and the referential observer (closest one)
    sorted_obs_time = sorted(obs_time.items(), key=operator.itemgetter(1))
    sorted_obs = [x[0] for x in sorted_obs_time]
    o1 = min(obs_time, key=obs_time.get)

    ### Gets the nodes of the graph and initializes likelihood
    nodes = np.array(list(graph.nodes))
    s_estimator = {}

    ### Print variables to be given to output to communicate intermediate results
    d_mu = collections.defaultdict(list)
    covariance = collections.defaultdict(list)

    ### Computes classes of nodes with same position with respect to all observers
    classes = tl.classes(path_lengths, sorted_obs)

    ### Iteration over all nodes per class

    for s in nodes:
        ### BFS tree
        #tree_s = nx.bfs_tree(graph, s)
        tree_s = likelihood_tree(paths, s, sorted_obs)
        #for (u, v) in tree_s.edges():
        #    print('(u, v) = ', u, ' ', v)
        #for o in sorted_obs:
        #    print('obs ', o)
        ### Covariance matrix
        cov_d_s = tl.cov_mat(tree_s, graph, paths, sorted_obs, s)
        #print('covariance')
        #print(cov_d_s)
        D_s = np.diag(np.diag(cov_d_s))
        D_s_inv = np.linalg.inv(D_s)
        ### vector -> difference between observation time and mean arrival time for observers
        w_s = tl.w_vector(sorted_obs_time, mu, paths, s, tree_s)
        I = np.ones((len(w_s)))
        #print('I ', I.shape)
        ### MLE of initial time t0
        t0_s = ((I.T @ D_s_inv @ w_s) / (I.T @ D_s_inv @ I))
        #print('t0_s ', t0_s)
        ### Auxilary variable to make equation simpler to write
        #print('SHAPES')
        #print('w_s ', w_s.shape)
        #print('t0 ', t0_s.shape)
        #print('... ', (w_s - (t0_s*I)).shape)
        #print('... ', (w_s - (t0_s*I)).T.shape)
        #print('... ', (t0_s*I).shape)
        #print('I ', I.shape)
        z_s = ((w_s - (t0_s*I)).T) @ D_s_inv @ (w_s - (t0_s*I))
        #print('z_s ', z_s)
        ### estimator for the source node
        #print('s_estimator ', len(sorted_obs)*np.log(z_s) + np.log(np.linalg.det(cov_d_s)))
        s_estimator[s] = len(sorted_obs)*np.log(z_s) + np.log(np.linalg.det(D_s))


    ### Find the nodes where the source estimator is the lowest
    posterior = posterior_from_logLH(s_estimator)
    #print('s_estimate')
    #print(s_estimator)
    optimal_source = min(posterior.values())
    #print('opt ', optimal_source)
    source_candidates = list()
    ### Finds nodes where the source is optimal
    for src, value in posterior.items():
        if np.isclose(value, optimal_source, atol= 1e-08):
            source_candidates.append(src)

    return source_candidates, posterior

#################################################### Helper methods for ml algo
def posterior_from_logLH(loglikelihood):
    """Computes and correct the bias associated with the loglikelihood operation.
    The output is a likelihood.

    Returns a dictionary: node -> posterior probability

    """
    bias = logsumexp(list(loglikelihood.values()))
    return dict((key, np.exp(value - bias))
            for key, value in loglikelihood.items())


def logLH_source_tree(mu_s, cov_d, obs, obs_time):
    """ Returns loglikelihood of node 's' being the source.
    For that, the probability of the observed time is computed in a tree where
    the current candidate is the source/root of the tree.

    - mu_s is the mean vector of Gaussian delays when s is the source
    - cov_d the covariance matrix for the tree
    - obs_time is a dictionary containing the observervations: observer --> time
    - obs is the ordered list of observers, i.e. obs[0] is the reference

    """
    assert len(obs) > 1

    ### Creates the vector for the infection times with respect to the referential observer
    obs_d = np.zeros((len(obs)-1, 1))

    ### Loops over all the observers (w/o first one (referential) and last one (computation constraint))
    #   Every time it computes the infection time with respect to the ref obs
    for l in range(1, len(obs)):
        obs_d[l-1] = obs_time[obs[l]] - obs_time[obs[0]]

    ### Computes the log of the gaussian probability of the observed time being possible
    exponent =  - (1/2 * (obs_d - mu_s).T.dot(np.linalg.inv(cov_d)).dot(obs_d -
            mu_s))
    denom = math.sqrt(((2*math.pi)**(len(obs_d)-1))*np.linalg.det(cov_d))

    return (exponent - np.log(denom))[0,0], obs_d - mu_s


def likelihood_tree(paths, s, obs):
    """Creates a BFS tree with only observers at its leaves.

    Returns a BFS tree
    """
    tree = nx.Graph()
    for o in obs:
        p = paths[o][s]
        tree.add_edges_from(zip(p[0:-1], p[1:]))
    return tree

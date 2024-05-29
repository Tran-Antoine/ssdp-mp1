import numpy as np
from scipy import signal
import scripts.sigutil as sigutil
import random
import math

def fixed_optimal_lms(x, d, K):
    """
    Computes a fixed non-adaptive filter using the optimal LMS algorithm. 
    This naive approach is very slow, hence this should probably only be used 
    for benchmarking purposes.
    Args:
        x: Signal of noise corrolated with the noise corrupting s[n]
        d: Signal s[n] but corrupted by noise
        K: Number of past samples
    Returns:
        a single value s'[n] approximating s[n]
    """
    Rx, rdx = sigutil.autocorrelate(x, d, K)
    f = np.linalg.solve(Rx, rdx)
    return f, d - np.convolve(x, f)[0:len(d)]

def fixed_iterative_lms(x, d, K, N_it):
    """
    Computes a fixed non-adaptive filter using the iterative LMS algorithm.
    This approach is faster than the optimal one, but might be less accurate.
    """
    Rx, rdx = sigutil.autocorrelate(x, d, K)
    eigenvalues, _ = np.linalg.eig(Rx)
    lambda_max = np.max(eigenvalues)
    lambda_min = np.min(eigenvalues)

    # Choose step size mu such that 0 < mu < 2/lambda_max
    mu = 2/(lambda_max + lambda_min)

    if mu >= 2/lambda_max:
        raise Exception("Step size cannot be bigger than 2/lambda_max!")
    
    f_it = np.zeros(K)

    for _ in range(N_it):
        f_it = f_it + mu * (rdx - Rx @ f_it)

    return f_it, d - np.convolve(x, f_it)[0:len(d)]

def adaptive_iterative_lms(x, d, K, N_it, algoType='LMS', mu=None, lambda_=None, delta=None, callback=None):
    """
    Computes an adaptive filter using the algoType algorithm. algoType is a string that allows
    the selection between three algorithms:
    - LMS: Standard LMS
    - NLMS: Normalized LMS (a type of nonlinear LMS)
    - RLS: Recursive LS (another type of nonlinear LMS)
    This takes an optional callback that can be used to plot things during the computation.
    """
    f_ad = np.zeros(K)
    e_ad = np.zeros(len(d))

    # Adaptive filtering
    for i in range(K, len(e_ad)):
        X = x[i-K:i]

        # Solve normal equation using algoType method
        if(algoType == 'LMS'):
            for _ in range(N_it):
                f_ad = f_ad + mu * X * (d[i] - f_ad.T @ X)

        elif(algoType == 'NLMS'):
            eps = 0.05

            for _ in range(N_it):
                normalizationFactor = mu/(eps + np.dot(X, X))
                f_ad = f_ad + normalizationFactor * X * (d[i] - f_ad.T @ X)

        elif(algoType == 'RLS'):
            omega = 1/delta * np.identity(K)
            
            for _ in range(N_it):
                z = omega @ X
                g = z / (lambda_ + np.dot(X, z))
                f_ad = f_ad + g * (d[i] - f_ad.T @ X)
                omega = 1/lambda_ * (omega - np.dot(g.reshape(-1, 1), z.reshape(1, -1)))

        else:
            raise ValueError('Invalid algo type ! Algo type must be either LMS or NLMS or RLS!')

        # Update output
        e_ad[i] = d[i] - f_ad.T @ X

        # Plot the iterative filter at some time instants (every 7 seconds starting from 16 seconds)
        if callback:
            callback(i, f_ad[::-1])

    f_ad = f_ad[::-1]

    return f_ad, e_ad

def set_position(i, vec, solution_space):
    solution_space[:, i] = vec

def set_random_position(i, K, solution_space):
    set_position(i, np.random.default_rng().uniform(-1, 1, K), solution_space)

def position(i, solution_space):
    return solution_space[:,i]

def neighbor(i, N_sols, solution_space):
    xik = position(i, solution_space)
    j = random.randint(0, N_sols - 2)
    if j == i: 
        j = N_sols - 1
    
    xjk = position(j, solution_space)

    return xik + (2*random.random()-1) * (xik - xjk)

def error_signal(f, X, idx, d):
    return d[idx] - f.T @ X
def error(f, X, idx, d):
    if f is None or len(f) == 0 or math.isnan(f[0]):
        return 10e20
    return (d[idx] - f.T @ X)**2

def potential_reward(f, X, idx, d):
    if f is None or len(f) == 0 or math.isnan(f[0]):
        return 0
    Ji = error(f, X, idx, d)
    return 1 / (1 + Ji)

def reward(i, X, idx, solution_space, d):
    fi = position(i, solution_space)
    return potential_reward(fi, X, idx, d)


def abc_one_sample(X, N_bees, iteration, solution_space, solution_space_tries, d, limit, K):
    N_sols = int(N_bees / 2)
    best_source = solution_space[:,0]  # will contain the values of the filter
    best_error = error(best_source, X, iteration, d)
    N_ITER = 20

    # TODO: find a better termination (like when it stops improving) instead of fixed number of steps
    for c in range(N_ITER):
        
        # if(c % 10 == 0):
        #     print(f"Completed {c}/{N_ITER}")
        # Each employee gets attributed a weight to indicate how good its source is
        weights = [reward(k, X, iteration, solution_space, d) for k in range(0, N_sols)]
        weights /= sum(weights)
        
        # Now all employees come back and communicate. Onlookers are going to pick the best sources
        # according to their weightages. Some sources might get multiple onlookers, others none
        solution_space_coverage = np.zeros(N_sols)

        for onlooker in range(N_bees):
            chosen_source = np.random.choice(N_sols, p=weights)
            solution_space_coverage[chosen_source] += 1

        # For each food source, compute a new neighboring position for each
        # onlooker bee that chose to go there  
        for (index, n) in enumerate(solution_space_coverage):
            current_max_reward = reward(index, X, iteration, solution_space, d)
            current_max_position = position(index, solution_space)
            success = False
            failures = 0
            for _ in range(int(n)):
                next_position = neighbor(index, N_sols, solution_space)
                next_reward = potential_reward(next_position, X, iteration, d)

                if next_reward > current_max_reward:
                    success = True
                    current_max_reward = next_reward
                    current_max_position = next_position
                else:
                    failures += 1

            # If at least one position was better, select the best one
            # and reset the number of tries
            if success:
                set_position(index, current_max_position, solution_space)
                solution_space_tries[index] = 0

                # If the location is the overall best (not just around this source), update it
                error_value = error(current_max_position, X, iteration, d)
                if error_value < best_error:
                    best_error = error_value
                    best_source = current_max_position

            elif failures == 0:
                # not being picked at all counts as a single failure
                solution_space_tries[index] += 1
            else:
                solution_space_tries[index] += failures

        # Employed bee with 'dead' locations go scout for other locations
        for (index, n) in enumerate(solution_space_tries):
            if n <= limit:
                continue
            set_random_position(index, K, solution_space)

    final_error = error_signal(best_source, X, iteration, d)
    if abs(final_error) > 1:
        return 0
    return final_error

def exec_task(i, slicing, x, K, limit, d, N_bees, solution_space, solution_space_tries):

    print(f"Running task with indices {i} to {min(i+slicing,len(d))}", flush=True)
    
    indices = []
    results = []

    for k in range(i, min(len(d), i+slicing)):
       
        X = x[k-K:k]
        out = abc_one_sample(X, N_bees, k, solution_space, solution_space_tries, d, limit, K)
        
        indices.append(k)
        results.append(out)

    print(f"Finished task with indices {i} to {min(i+slicing,len(d))}", flush=True)
    return indices, results
def adaptive_abc(x, d, K, N_bees, limit):
    """
    Computes an adaptive filter using the improved ABC algorithm for adaptive filtering
    """
    N_sols = int(N_bees / 2)

    # Start with random food sources
    solution_space = np.random.default_rng().uniform(-0.2, 0.2, [K, N_sols])
    # Counts for each food source how many times we tried to improve it.
    # Whenever we go above `limit`, it means that it's time to abandon the source
    solution_space_tries = np.zeros(N_sols)



    #f_ad = np.zeros(K)
    e_ad = np.zeros(len(d))

    from functools import partial
    from concurrent.futures import ProcessPoolExecutor

    SLICING = 10000

    partial_exec_task = partial(exec_task, slicing=SLICING, x=x, K=K, limit=limit, d=d, N_bees=N_bees, solution_space=solution_space, solution_space_tries=solution_space_tries)
        
    with ProcessPoolExecutor(16) as executor:
        for idx_range, results in executor.map(partial_exec_task, range(K, len(e_ad), SLICING)):
            for i, r in zip(idx_range, results):
                e_ad[i] = r

    """
    for i in range(K, len(e_ad)):
        _, res = exec_task(i, x, K, limit, d, N_bees, solution_space, solution_space_tries)
        e_ad[i] = res
    """

    return e_ad
    """

    # Adaptive filtering
    for i in range(K, len(e_ad)):
        X = x[i-K:i]
        e_ad[i] = abc_one_sample(X, i)
        if i%5000 == 0:
            print(f'{(i-K)/(len(e_ad)-K)}')

    return e_ad
    """
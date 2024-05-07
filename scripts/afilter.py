import numpy as np
from scipy import signal
import scripts.sigutil as sigutil

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

def adaptive_abc(x, d, K, N_bees, limit):
    import random
    import math
    """
    Computes an adaptive filter using the improved ABC algorithm for adaptive filtering
    """
    N_sols = N_bees / 2

    # Start with random food sources
    solution_space = np.random.default_rng().uniform(-1, 1, [K, N_sols])
    # Counts for each food source how many times we tried to improve it.
    # Whenever we go above `limit`, it means that it's time to abandon the source
    solution_space_tries = np.zeros(N_sols)

    def set_position(i, vec):
        solution_space[:, i] = vec

    def set_random_position(i):
        set_position(i, np.random.default_rng().uniform(-1, 1, K))

    def position(i):
        return solution_space[:,i]
    
    def neighbor(i):
        xik = position(i)
        j = random.randint(0, N_sols - 2)
        if j == i: 
            j = N_bees - 1
        
        xjk = position(j)

        return xik + (2*random.random()-1) * (xik - xjk)
    
    def reward(i):
        fi = position(i)
        return potential_reward(fi)
    
    def potential_reward(vec):
        if not vec or len(vec) == 0 or math.isnan(vec[0]):
            return 0.0
        Ji = d - np.convolve(x, vec)[0:len(d)]
        return 1 / (1 + Ji*Ji)
    
    def p(i):
        reward_value = reward(i)
        norm = sum([reward(k) for k in range(0, N_sols)])
        return reward_value / norm
    
    # Each employee goes to their random food source, finds a neighboring one
    # and moves there if it's a better source (otherwise does nothing)
    for employee in range(N_sols):
        current_reward = reward(employee)
        next_position = neighbor(employee)
        next_reward = potential_reward(next_position)

        # if the next source is better, go there and forget about the current source
        if next_reward > current_reward:
            solution_space[:,employee] = next_position
            solution_space_tries[employee] = 0 # reset the number of tries
        else:
            solution_space_tries[employee] += 1 # increment try counter

    while True:
        # Each employee gets attributed a weight to indicate how good its source is
        weights = [reward(k) for k in range(0, N_sols)]
        weights /= sum(weights)

        # Now all employees come back. Onlookers are going to pick the best sources
        # according to their weightages. Some sources might get multiple onlookers, others none
        solution_space_coverage = np.zeros(N_bees)

        for onlooker in range(N_sols):
            chosen_source = np.random.choice(N_bees, p=weights)
            solution_space_coverage[chosen_source] += 1

        # For each food source, compute a new neighboring position for each
        # onlooker bee that chose to go there        
        for (index, n) in enumerate(solution_space_coverage):
            current_max_reward = reward(index)
            current_max_position = position(index)
            success = False
            failures = 0
            for _ in range(n):
                next_position = neighbor(index)
                next_reward = potential_reward(next_position)

                if next_reward > current_max_reward:
                    success = True
                    current_max_reward = next_reward
                    current_max_position = next_position
                else:
                    failures += 1

            # If at least one position was better, select the best one
            # and reset the number of tries
            if success:
                set_position(index, current_max_position)
                solution_space_tries[index] = 0
            elif failures == 0:
                # not being picked at all counts as a single failure
                solution_space_tries[index] += 1
            else:
                solution_space_tries[index] += failures

        # Employed bee with 'dead' locations go scout for other locations
        for (index, n) in enumerate(solution_space_tries):
            if n <= limit:
                continue
            set_random_position(index)
            
    
        


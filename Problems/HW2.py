import numpy as np
import cvxpy as cp
from typing import List
import itertools
from scipy.optimize import linear_sum_assignment

"""
docs partially generated with the help of github copilot
-Chun Kai, 2024
"""

def generate_random_powers(n: int, seed=None):
    """
    Generate two lists of random, non-repeating integers representing powers of each horse, one for each player.

    This function generates two lists of length `n` each, containing unique integers 
    from the range [0, 2*n). The integers are shuffled randomly to ensure that the 
    lists are different each time the function is called, unless a specific seed is provided.

    Parameters:
        n (int): The number of elements in each list.
        seed (int, optional): An optional seed for the random number generator to ensure 
                              reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing two lists of integers. Each list has `n` unique integers.
    """
    rng = np.random.default_rng(seed=seed)
    powers_all = list(range(n*2))
    rng.shuffle(powers_all)
    powers1 = powers_all[:n]
    powers2 = powers_all[n:]
    return powers1, powers2

def full_horse_racing_matrix(q: int, power1=None, power2=None, values=None):
    """
    Generates a matrix representing the outcomes of horse races between two players with given horse power levels.

    Parameters:
        q (int): The number of horses each player has.
        power1 (array-like, optional): The power levels of Player 1's horses. Defaults to range(q).
        power2 (array-like, optional): The power levels of Player 2's horses. Defaults to range(q).
        values (array-like, optional): The values assigned to each race. Defaults to range(1, q+1).

    Returns:
    tuple: A tuple containing:
        - M (numpy.ndarray): A matrix of shape (n, n) 
                             Each element M[i, j] represents the outcome of the race between the i-th permutation of Player 1's horses 
                             and the j-th permutation of Player 2's horses.
        - acts1 (list): A list of all permutations of Player 1's horses (i.e., all possible assignments)
            That is, acts1[i][loc] is the power of the horse (not the index) at race loc fo player 1 assuming it chooses action i
        - acts2 (list): A list of all permutations of Player 2's horses (i.e., all possible assignments).
    """
    if power1 is None:
        power1 = np.array(range(q))
    if power2 is None:
        power2 = np.array(range(q))
    if values is None:
        values = np.array(range(q)) + 1  # make values 1,2,...q

    acts1 = list(itertools.permutations(power1, q))
    acts2 = list(itertools.permutations(power2, q))

    n, m = len(acts1), len(acts2)
    M = np.zeros((n, m))

    for loc in range(q):
        for i in range(n):
            for j in range(m):
                if acts1[i][loc] > acts2[j][loc]:  # P1 wins, so add values (note difference from HW1)
                    M[i, j] += values[loc]
                elif acts1[i][loc] < acts2[j][loc]:  # P2 wins, so subtract values (note difference from HW1)
                    M[i, j] += -values[loc]

    return M, acts1, acts2

def subgame_matrix_from_scratch(powers1, powers2, actions1, actions2, values=None):
    """
    Compute the full subgame matrix from scratch where the row player maximizes their score.

    Parameters:
        powers1 (list): A list of power values for the first player.
        powers2 (list): A list of power values for the second player.
        actions1 (list of lists): A list of actions for the first player. 
                Each action1[i] is such that action1[i][j] is the horse index for battlefield j for player 1 for action_index i.
        actions2 (list of lists): A list of actions for the second player, where each action is a list of indices corresponding to powers2.
                Each action2[i] is such that action2[i][j] is the horse index for battlefield j for player 2 for action_index i.
        values (list, optional): A list of values for each race. If None, defaults to 1,...n
                
    Returns:
        numpy.ndarray: A matrix where each element represents the outcome of the subgame for the corresponding actions of the two players.
                    Positive values indicate a win for the first player, negative values indicate a win for the second player, 
                    and zero indicates a tie.
    """
    n, k = len(powers1), len(powers2)
    j, k = len(actions1), len(actions2)

    if values is None:
        values = np.array(range(n)) + 1  # make values 1,2,...n

    M = np.zeros((j, k))
    for action1_id, action1 in enumerate(actions1):
        for action2_id, action2 in enumerate(actions2):
            for race_id in range(n):
                if powers1[action1[race_id]] > powers2[action2[race_id]]:
                    M[action1_id, action2_id] += values[race_id]
                elif powers2[action2[race_id]] > powers1[action1[race_id]]:
                    M[action1_id, action2_id] -= values[race_id]
    return M

def update_subgame_matrix(M_old, powers1, powers2, actions1, actions2, new_action1, new_action2, values=None, neg=True):
    """
    Updates the subgame matrix with one new action per player and returns a subgame payoff matrix of size 1 larger in each dimension.
    Only recomputes the O(n) entries for the actions that were added and reuses the old matrix for the rest.

    Parameters:
        M_old (numpy.ndarray): The existing subgame matrix.
        powers1 (list): List of horse power values for the first player.
        powers2 (list): List of horse power values for the second player.
        actions1 (list of lists): List of actions taken by the first player.
        actions2 (list of lists): List of actions taken by the second player.
        new_action1 (list): New action taken by the first player.
        new_action2 (list): New action taken by the second player.
        values (list, optional): A list of values for each race. If None, defaults to 1,...n.
        neg (bool, optional): If True, negative values are used for losses; otherwise, positive values are used. Default is True.

    Returns:
        numpy.ndarray: The updated subgame matrix with the new actions incorporated.
    """
    n, k = len(powers1), len(powers2)
    j, k = len(actions1), len(actions2)

    if values is None:
        values = np.array(range(n)) + 1  # make values 1,2,...n

    M = np.zeros((j+1, k+1))
    M[:-1, :-1] = M_old

    if neg:
        neg = -1
    else:
        neg = 1

    for action2_id, action2 in enumerate(actions2):
        for race_id in range(n):
            if powers1[new_action1[race_id]] > powers2[action2[race_id]]:
                M[-1, action2_id] += neg * values[race_id]
            elif powers2[action2[race_id]] > powers1[new_action1[race_id]]:
                M[-1, action2_id] -= neg * values[race_id]

    for action1_id, action1 in enumerate(actions1):
        for race_id in range(n):
            if powers2[new_action2[race_id]] > powers1[action1[race_id]]:
                M[action1_id, -1] -= neg * values[race_id]
            elif powers1[action1[race_id]] > powers2[new_action2[race_id]]:
                M[action1_id, -1] += neg * values[race_id]

    for race_id in range(n):
        if powers1[new_action1[race_id]] > powers2[new_action2[race_id]]:
            M[-1,-1] += neg * values[race_id]
        elif powers1[new_action1[race_id]] < powers2[new_action2[race_id]]:
            M[-1,-1] -= neg * values[race_id]

    return M


def solve_ne_via_lp(P: np.ndarray):
    """
    Solves for the Nash Equilibrium (NE) of a two-player zero-sum game using linear programming.

    Parameters:
    P (np.ndarray): A 2D numpy array representing the payoff matrix for the row player. 
                    The row player aims to minimize the payoff, while the column player aims to maximize it.

    Returns:
    tuple: A tuple containing:
        - prob.value (float): The optimal value of the objective function (the game's value).
        - x.value (np.ndarray): The optimal mixed strategy for the row player.
        - constrs[0].dual_value (np.ndarray): The optimal mixed strategy for the column player (dual values).
    """
    # By default, the row player MINIMIZES
    n, m = P.shape

    x = cp.Variable(n, nonneg=True)
    V = cp.Variable(1)

    constrs = [P.T@x <= V, cp.sum(x) == 1]

    prob = cp.Problem(cp.Minimize(V), constrs)
    prob.solve(solver=cp.GUROBI) # you can use other solvers here, e.g., cp.ECOS

    return prob.value, x.value, constrs[0].dual_value

def double_oracle_solver(powers1, powers2, values, max_iter=30, eps=1e-3):
    """
    Solves a two-player zero-sum game using the double oracle algorithm.
    Parameters:
        powers1 (list): List of horse power values for player 1.
        powers2 (list): List of horse power values for player 2.
        values (list, optional): A list of values for each race. If None, defaults to 1,...n.
        max_iter (int, optional): Maximum number of iterations to run the algorithm. Default is 30.
        eps (float, optional): Convergence threshold. Default is 1e-3.
    Returns:
    tuple: A tuple containing:
        - game_val_row_player (float): The game value for the row player.
        - actions1 (list of lists): List of actions for player 1. 
            Each action is represented by a list of horse indexes for each location, i.e.,
            actions1[i] contains the horse index for race i. Note that in the full game solver
            we returned horse powers, not indices (as we do here).
        - actions2 (list of lists): List of actions for player 2.
        - ne1 (numpy.ndarray): Nash equilibrium strategy for player 1.
        - ne2 (numpy.ndarray): Nash equilibrium strategy for player 2.
    """
    assert len(powers1) == len(powers2)
    n = len(powers1)
    if values is None:
        values = np.array(range(n)) + 1  # make values 1,2,...n

    actions1 = [np.array(range(n))]
    actions2 = [np.array(range(n))]

    rewards1 = [reward_vector_given_action2(n, powers1, powers2, actions2[0], values)]
    rewards2 = [reward_vector_given_action2(n, powers2, powers1, actions1[0], values)]
    P = -subgame_matrix_from_scratch(powers1, powers2, actions1, actions2, values=values)

    for iter in range(max_iter):
        print('Iteration:', iter)
        # Compute NE of subgame given by actions1, actions2.
        # Note that the subgame matrix is negated because the row player maximizes whil the solver assumes row player minimizes.
        print('Computing subgame matrix')
        P = update_subgame_matrix(P, powers1, powers2, actions1[:-1], actions2[:-1], actions1[-1], actions2[-1], values=values, neg=True)

        print('Solving subgame')
        game_val, ne1, ne2 = solve_ne_via_lp(P) # Note game_val is for the minimizing row player ... 
        game_val_row_player, game_val_col_player = -game_val, game_val # so we have to reverse the sign here.

        print('Computing best responses')
        br1, val1 = br(rewards1, ne2, method='MWB')
        br2, val2 = br(rewards2, ne1, method='MWB')
        
        # Uncomment if you'd like to track performance over iterations.
        # print(val1, game_val_row_player, val2, game_val_col_player)

        # Termination criterion
        if val1 <= game_val_row_player + eps and val2 <= game_val_col_player + eps:
            break

        # Add new actions to the list, alongside their reward vectors.
        actions1.append(br1)
        actions2.append(br2)

        rewards1.append(reward_vector_given_action2(n, powers1, powers2, actions2[-1], values))
        rewards2.append(reward_vector_given_action2(n, powers2, powers1, actions1[-1], values))

    print(f'Completed in {iter} iterations')
    return game_val_row_player, actions1, actions2, ne1, ne2

def br(rewards: List[np.ndarray], probs: List[float], method='LP'):
    """
    Compute the best response given probabilities of each action by the opponent player, as well
    as the rewards (weights in a complete bipartite graph) for each of those actions.
    Parameters:
        rewards (list of ndarrays of sizes n*n): A list of reward arrays, one for each action of the player. 
            Each reward array is a 2D numpy array representing the rewards for each action, where
            rewards[i][j, k] containing the expected reward if the opponent plays action i and we match horse j to race k.
        probs (list of floats): A list or array of probabilities corresponding to the rewards.
        method (str, optional): The method to use for computing the maximum weight bipartite matching. 
                                Default is 'LP' (Linear Programming). Can also use specialized 
                                'MWB' (Maximum Weight Bipartite Matching) algorithms.
    Returns:
    tuple: A tuple containing:
        - matching (ndarray of size n): An integer array representing the matching. 
            matching[i] contains the index of the horse matched to race i.
        - value (float): The value of the maximum weight bipartite matching.
    """
    W = get_bipartite_weights(rewards, probs)
    matching, value = max_weight_bipartite_matching(W, method=method)
    
    # convert matching to *integer* array
    matching = matching.astype(int)

    return matching, value

def max_weight_bipartite_matching(W, method='LP'):
    assert W.shape[0] == W.shape[1]
    n = W.shape[0]

    # Use LP
    if method == 'LP':
        x = cp.Variable((n, n), nonneg=True) # x[i, j] is 1 iif i is matched to j

        constrs = [x <= 1.0, 
                   cp.sum(x, axis=0) == np.ones(n), 
                   cp.sum(x, axis=1)==np.ones(n)]

        prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(x, W))), constrs)
        prob.solve(solver=cp.GUROBI)

        # Extract matching from answer
        M = [0] * n
        for race_id in range(n):
            for horse_id in range(n):
                if np.isclose(x.value[horse_id][race_id], 1.0):
                    M[race_id] = horse_id
        
        return np.array(M, int), prob.value

    # Use maximum weight bipartite matching
    elif method == 'MWB':
        # We use W.T for our weights (matching races to horses)
        # to recover the solution a bit more easily.
        # For this function, row indices are races, column indices are horses.
        row_ind, col_ind = linear_sum_assignment(W.T, maximize=True)
        assert np.all(row_ind == np.array(range(n)))
        return col_ind, W.T[row_ind, col_ind].sum()


def get_bipartite_weights(reward_vectors2, probabilities):
    """
    Take weighted sum of reward vectors (weighed by probabilities).
    """
    assert len(reward_vectors2) == len(probabilities)
    assert np.isclose(np.sum(probabilities), 1.0)
    assert np.all(probabilities >= -1e6)

    size_strategy_set = len(reward_vectors2)

    W = sum([reward_vectors2[i] * probabilities[i] for i in range(size_strategy_set)])
    return W

def reward_vector_given_action2(n, power1, power2, action2, values):
    """
    Compute the reward vector for the first for a given the second player's action.
    """
    M = np.zeros((n, n))
    for horse1_id in range(n):
        for race_id in range(n):
            # print(power1[horse1_id], power2[action2[race_id]])
            if power1[horse1_id] > power2[action2[race_id]]:
                M[horse1_id][race_id] = values[race_id]
            elif power1[horse1_id] < power2[action2[race_id]]:
                M[horse1_id][race_id] = -values[race_id]
    return M        

def solve_via_double_oracle(powers1, powers2, values, max_iter, eps):
    """
    Solves a game using the double oracle algorithm.

    Parameters:
    powers1 (list of floats): List of power values for the first player.
    powers2 (list of floats): List of power values for the second player.
    values (list of floats): List of values representing the payoff matrix.
    max_iter (int): Maximum number of iterations for the double oracle algorithm.
    eps (float): Convergence threshold for the double oracle algorithm. 
        Is slower, but gets a more accurate solution if set to a smaller value.

    Returns:
    tuple: A tuple containing:
        - val_do (float): The value of the game, also the expected utility for the first player under eqm.
        - actions1 (list): List of lists, each containing the assignment of horse-indices to locations.
        - actions2 (list): List of lists, each containing the assignment of horse-indices to locations.
        - ne1 (list): Nash equilibrium strategies over the actions in actions1.
        - ne2 (list): Nash equilibrium strategies over the actions in actions2.
    """
    val_do, actions1, actions2, ne1, ne2 = double_oracle_solver(powers1, powers2, values, max_iter=1000, eps=1e-3)

    return val_do, actions1, actions2, ne1, ne2

def solve_via_enumeration(powers1, powers2, values):
    """
    Solves a horse racing problem via enumeration.

    This function computes the Nash equilibrium for a horse racing game using enumeration.
    It first constructs the full horse racing matrix (which is exponential in n), then solves for the Nash equilibrium
    using linear programming.

    Parameters:
        powers1 (list of floats): A list of power values for the first set of horses.
        powers2 (list of floats): A list of power values for the second set of horses.
        values (list of floats): A list of values associated with the horse races.

    Returns:
    tuple: A tuple containing:
        - val_lp (float): The value of the linear programming solution giving the payoffs to P1 under the NE.
        - all_acts1_horse_powers (list): List of lists, each containing the assignment of horse-powers to locations.
        - all_acts2_horse_powers (list): List of lists, each containing the assignment of horse-powers to locations.
        - ne1 (list): The Nash equilibrium strategy over the actions in acts1_horse_powers.
        - ne2 (list): The Nash equilibrium strategy over the actions in acts2_horse_powers.
    """
    M, all_acts1_horse_powers, all_acts2_horse_powers = full_horse_racing_matrix(n, powers1, powers2, values=values)
    val_lp, ne1, ne2 = solve_ne_via_lp(-M)
    val_lp = -val_lp # need to negate this value because we are solving the negated problem

    return val_lp, all_acts1_horse_powers, all_acts2_horse_powers, ne1, ne2

def sanity_checks():
    # Sanity check where P1 always beats P2
    n=3
    powers1 = [10, 11, 12]
    powers2 = [1,2,3]
    values = [1,2,3]
    ENUM_val_enum, ENUM_acts1_horse_powers, ENUM_acts2_horse_powers, ENUM_ne1, ENUM_ne2, = solve_via_enumeration(powers1, powers2, values)
    DO_val, DO_acts1_horse_indices, DO_acts1_horse_indices, DO_ne1, DO_ne2 = solve_via_double_oracle(powers1, powers2, values, max_iter=1000, eps=1e-3)
    print("Double Oracle Value:", DO_val, "Enumeration Value:", ENUM_val_enum)
    # expected value = 6

    # Sanity check where P2 always beats P1
    n=3
    powers2 = [10, 11, 12]
    powers1 = [1,2,3]
    values = [1,2,3]
    ENUM_val_enum, ENUM_acts1_horse_powers, ENUM_acts2_horse_powers, ENUM_ne1, ENUM_ne2, = solve_via_enumeration(powers1, powers2, values)
    DO_val, DO_acts1_horse_indices, DO_acts1_horse_indices, DO_ne1, DO_ne2 = solve_via_double_oracle(powers1, powers2, values, max_iter=1000, eps=1e-3)
    print("Double Oracle Value:", DO_val, "Enumeration Value:", ENUM_val_enum)
    # expected value = -6

    # Sanity check where P1 always loses to P2 save for one powerhorse
    n=3
    powers2 = [10, 11, 12]
    powers1 = [1,2,100]
    values = [1,2,5]
    ENUM_val_enum, ENUM_acts1_horse_powers, ENUM_acts2_horse_powers, ENUM_ne1, ENUM_ne2, = solve_via_enumeration(powers1, powers2, values)
    DO_val, DO_acts1_horse_indices, DO_acts1_horse_indices, DO_ne1, DO_ne2 = solve_via_double_oracle(powers1, powers2, values, max_iter=1000, eps=1e-3)
    print("Double Oracle Value:", DO_val, "Enumeration Value:", ENUM_val_enum)
    # expected value = 2

if __name__ == '__main__':
    # sanity_checks()

    print('Performing experiments on small instances')
    n = 5
    powers1, powers2=generate_random_powers(n, seed=9)
    values = np.array(range(n)) + 1  # make values 1,2,...n
    # values = np.ones(n)

    DO_val, DO_acts1_horse_indices, DO_acts1_horse_indices, DO_ne1, DO_ne2 = solve_via_double_oracle(powers1, powers2, values, max_iter=1000, eps=1e-3)
    ENUM_val_enum, ENUM_acts1_horse_powers, ENUM_acts2_horse_powers, ENUM_ne1, ENUM_ne2, = solve_via_enumeration(powers1, powers2, values)

    print("Double Oracle Value:", DO_val, "Enumeration Value:", ENUM_val_enum)

    print('Testing on larger instance')
    n = 20
    powers1, powers2=generate_random_powers(n, seed=10)
    values = np.array(range(n)) + 1  # make values 1,2,...n
    DO_val, DO_acts1_horse_indices, DO_acts1_horse_indices, DO_ne1, DO_ne2 = solve_via_double_oracle(powers1, powers2, values, max_iter=1000, eps=1e-3)
    # Try using enumeration to solve the same problem here?
    print("Double Oracle Value:", DO_val)
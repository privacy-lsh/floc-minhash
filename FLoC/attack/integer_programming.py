import cvxpy as cp
import numpy as np

import FLoC.utils as utils
from FLoC.chromium_components import sim_hash
import gurobipy as gp
from gurobipy import GRB
import logging
from FLoC.utils import OldTimer # replaced with more complete code where took inspiration from in codetiming
from codetiming import Timer
from pip._vendor.colorama import Fore, init # colorama always in pip._vendor.colorama
init(autoreset=True) # to avoid reseting color everytime

##################################################
## Function related to solving integer programs ##
##################################################

# Solve integer program
def solve_ip_cvxpy(hash_history, output_hash_bitcount, r_target_bits, multiple_solution=True):
    """
    Legacy first attempt at solving SimHash problem with integer programming using cvxpy solver.
    :param hash_history: the hash of the movie title in the input history
    :param output_hash_bitcount: output length of the target SimHash
    :param r_target_bits: the SimHash target bits (in reversed order) in an iterable (e.g. string)
    :param multiple_solution: if attempt to find multiple solution to the same SimHash problem
    :return: the SimHash problems and variables
    """
    # [1] https://www.cvxpy.org/tutorial/intro/index.html#vectors-and-matrices
    # Constraints
    # sum gaussian sample > 0 if 1 < 0 if 0
    matrix = [[0.0] * len(hash_history) for _ in range(output_hash_bitcount)]
    for dimension in range(output_hash_bitcount):
        # Determine sign of inequality for constraint
        # Note can be different from simhash on equality to 0 for the constraint
        if int(r_target_bits[dimension]) == 0:
            sign_weight = 1
        else:
            sign_weight = -1
        for i, hash in enumerate(hash_history, start=0):
            matrix[dimension][i] = sim_hash.random_gaussian(dimension, hash) * sign_weight

    Gaussian_coordinates = np.array(matrix)
    # [2] https://www.cvxpy.org/tutorial/advanced/index.html
    subset_selection = cp.Variable(shape=(len(hash_history), 1), boolean=True)
    # @ for matmul inequality constraint are elementwise in cvxpy
    simhash_constraint = Gaussian_coordinates @ subset_selection <= 0  # Strict inequalities are not allowed
    # Define an objective: try to maximize the size of the subset
    # As 0 is always a trivial solution minimize would not be suitable
    objective = cp.Maximize(cp.sum(subset_selection))

    simhash_problem = cp.Problem(objective=objective, constraints=[simhash_constraint])
    logging.debug(f'Solving Integer (boolean) Program')

    with Timer(name='solve_simhash', text=f"{Fore.LIGHTMAGENTA_EX}Time spent solving Integer Program {{:.5f}}", logger=logging.warning):
        simhash_problem.solve() # can specify different solvers etc

    logging.debug(f'Results:')
    logging.debug(f'status: {simhash_problem.status}')
    logging.info(f'optimal value: {simhash_problem.value}')
    logging.debug(f'optimal variable: {subset_selection.value}')
    logging.info(f'solve time: {simhash_problem.solver_stats.solve_time}')

    # return simhash_problem, subset_selection
    if not multiple_solution:
        return [simhash_problem], [subset_selection]
    problems = [simhash_problem]
    variables = [subset_selection]
    # Use named tuple for code compatibility with implementation already existing if had side effect or others
    # SimhashProblem = namedtuple('SimhashProblem', ['status', 'value', 'solver_stats'])
    # problems = [SimhashProblem(simhash_problem.status, simhash_problem.value, simhash_problem.solver_stats)]
    # Variable = namedtuple('Variable', ['value'])
    # variables = [Variable(subset_selection.value)]

    # Try to find more solutions
    # Defines a new problem where only add one more constraints at each iteration
    while simhash_problem.value != 0:
        subset_selection = cp.Variable(shape=(len(hash_history), 1), boolean=True) # redefine cause need to store solution ?
        simhash_constraint = Gaussian_coordinates @ subset_selection <= 0 # gaussian_coord matrix does not change ?
        objective = cp.Maximize(cp.sum(subset_selection))
        constraints_list = [simhash_constraint]
        # Tried to add constraint so that output solution should be different from last ones
        # for prev_vars in variables:
            # [1] https://www.cvxpy.org/tutorial/intro/index.html#constraints
            # [1] does not say != is allowed also error for chain constraints
            # constraints_list.append(subset_selection != prev_vars.value) # cannot use this constraint as is
        # Here try to reduce
        max_value = problems[-1].value - 1
        constraints_list.append(cp.sum(subset_selection) <= max_value)
        simhash_problem = cp.Problem(objective=objective, constraints=constraints_list)
        with Timer(name='solve_simhash'):
            simhash_problem.solve()
        logging.debug(f'optimal value: {simhash_problem.value}\nstatus: {simhash_problem.status}\n{subset_selection.value}')
        if simhash_problem.value != 0:
            problems.append(simhash_problem)
            variables.append(subset_selection)

    return problems, variables


# Optimal value should be 2D array with 1 column and as much rows as movie_id_hist length
def reconstruct_sol(optimal_vars, movie_id_hist):
    """
    Helper function to reconstruct the history subset from the (cvxpy) integer program solution
    :param optimal_vars: the boolean variable values corresponding to the optimal solution
    :param movie_id_hist: the input movieID history
    :return: the subset of the movieID history matching the target SimHash
    """
    selected_history_subset = []
    counter = 0
    for row in optimal_vars:
        is_id_selected = row[0]
        if is_id_selected == 1.:
            selected_history_subset.append(movie_id_hist[counter])
        # else is_id_selected should be 0.
        counter += 1
    return selected_history_subset


# counter_stats side effect modify value referenced
def solve_ip_gurobi(hash_history, output_hash_bitcount, r_target_bits, movie_id_hist, PARAMETERS, counter_stats=None, use_matrices=True,
                    max_sol_count=1024, print_ip=False):
    '''
    Solve SimHash problem with integer programming using gurobi solver.
    :param hash_history: hashes of the movie titles in the input movie history
    from which wants to find subset matching the target SimHash
    :param output_hash_bitcount: how many bits of the target SimHash should match
    :param r_target_bits: the reversed target SimHash bits
    :param movie_id_hist: movieIDs of the input movie history for which want to find a subset matching target simhash
    (used for retrieving the subset movie history)
    :param PARAMETERS: need 'find_mult_sol', 'gurobi_timelimit' and 'timer_nametag' in a dict
    :param counter_stats: if want to compute some statistics in a Counter
    :param use_matrices: if use matrices instead of list of vectors
    :param max_sol_count: maximum number of solutions to look for when look for multiple (check doc. for changes, max value: 2000000000)
    :param print_ip: if asks gurobi to print the integer program
    :return: The SimHash integer program model, tupled with the optimal variable and the selected history subset
    '''
    # To remove more prints can check [] https://support.gurobi.com/hc/en-us/articles/360044784552-How-do-I-suppress-all-console-output-from-Gurobi-
    # Create a new model
    simhash_model = gp.Model("simhash ip")

    if not print_ip:
        # Some params can already be defined in gurobi.env and changed after here too
        # Set logging verbosity:
        # see [1] on how to set params
        # [1] https://www.gurobi.com/documentation/9.1/refman/python_parameter_examples.html#PythonParameterExamples
        # Doc of setParam says to use `Model.setParam() to change parameter settings for an existing model.`
        # shut off all logging (when 0) default 1
        # simhash_model.Params.outputflag = 0 # ignore case and underscore when call like this
        # gp.Model.setParam(simhash_model, "OutputFlag", 0)
        simhash_model.setParam(GRB.Param.OutputFlag, 0)
        # If want more granularity (with those two still log to console and file with logging)
        simhash_model.Params.logfile = '' # no log file
        simhash_model.Params.logtoconsole = 0

    # Set timeLimit (also exist TuneTimeLimit param)
    simhash_model.Params.timelimit = PARAMETERS['gurobi_timelimit'] # 10 # in seconds

    # Create variables
    # as a list of variable:
    if use_matrices:
        # as a vector of variable
        subset_selection = simhash_model.addMVar(len(hash_history), vtype=GRB.BINARY, name="subset_selection")
    else:
        subset_selection = simhash_model.addVars(len(hash_history), vtype=GRB.BINARY, name="subset_selection")


    # Set objective
    # could try with different objectives,
    if use_matrices:
        # could try MLinExpr which is returned by .sum()
        simhash_model.setObjective(subset_selection.sum(), GRB.MAXIMIZE) # can also use range like x[:,1].sum()
    else:
        # always considered obj as 0 cause m.getVars() returns 0 ? unless it is called after optimize ?
        simhash_model.setObjective(gp.quicksum(subset_selection), GRB.MAXIMIZE)


    # Constraints
    # sum gaussian sample > 0 if 1 < 0 if 0
    Gaussian_coordinates = [[0.0] * len(hash_history) for _ in range(output_hash_bitcount)]
    for dimension in range(output_hash_bitcount):
        # Determine sign of inequality for constraint
        # can be different from simhash on equality to 0 for the constraint
        if int(r_target_bits[dimension]) == 0:
            sign_weight = 1
        else:
            sign_weight = -1
        for i, hash in enumerate(hash_history, start=0):
            Gaussian_coordinates[dimension][i] = sim_hash.random_gaussian(dimension, hash) * sign_weight

    if use_matrices:
        Gaussian_coordinates = np.array(Gaussian_coordinates) # do not need ndarray ?
        simhash_model.addConstr(Gaussian_coordinates @ subset_selection <= 0, "c0")
    else:
        # changed < 0 to <= 0 following error similar to [1] https://groups.google.com/g/gurobi/c/sXM6WEciljk?pli=1
        # reasoning mentionned for not implemented: ">0" is difficult to optimize since the ideal result would be infinitesimally close to zero, but not zero.
        for dim in range(output_hash_bitcount):
            # need to add one constraint per row of the matrix
            simhash_model.addConstr(gp.quicksum(Gaussian_coordinates[dim][i] * subset_selection[i] for i in range(len(hash_history)))  <= 0, f"c{dim}")

    # test print
    # print(f'Test print')
    # obj = m.getObjective()
    # print(f'obj:\n{obj.getValue()}')
    # did not print anything:
    # for i in range(obj.size()):  # not including constant
    #     print(f'{obj.getVar(i), obj.getCoeff(i), obj.getConstant(i)}')

    # print(f'cons:\n{m.getConstrs()}')
    # m.printAttr('subset_selection')

    # try to find multiple solution [0] and code example [1]
    # [0] https://www.gurobi.com/documentation/9.1/refman/finding_multiple_solutions.html
    # [1] https://www.gurobi.com/documentation/9.1/examples/poolsearch_py.html#subsubsection:poolsearch.py
    if PARAMETERS['find_mult_sol']:
        # Limit how many solutions to collect Max value: 2000000000
        # power set (size: 2^n) give number of subset of a set of cardinality n
        # [2] https://www.gurobi.com/documentation/9.1/refman/poolsolutions.html
        simhash_model.setParam(GRB.Param.PoolSolutions, max_sol_count)
        # Limit the search space by setting a gap for the worst possible solution that will be accepted
        # simhash_model.setParam(GRB.Param.PoolGap, 0.10)
        # do a systematic search for the k-best solutions
        simhash_model.setParam(GRB.Param.PoolSearchMode, 2)
        # some subtleties in [3]
        # [3] https://www.gurobi.com/documentation/9.5/refman/subtleties_and_limitations.html
        # Say to set optimality gap to zero when using PoolSearchMode=2
        # this would mean setting:MIPGap and MIPGapAbs to zero ?

    # Optimize model
    # with Timer(f"solve_simhash gp multi_sol: {PARAMETERS['find_mult_sol']}"):
    # timer_nametag used when evaluating runtimes and solver is run with different parameters for generated histories
    # in the same run.
    with Timer(name=f"gsolve_ip_{PARAMETERS['timer_nametag']}_multi_sol_{PARAMETERS['find_mult_sol']}", text=f"{Fore.LIGHTMAGENTA_EX}Time spent solving Integer Program {{:.5f}}",
               logger=logging.d5bg): # logging.warning
        simhash_model.optimize()

    # For visualization this is better (or maybe it was default visualization from default gurobi logger)
    if print_ip:
        # print(simhash_model.display()) # need to be called after optimize()
        # Note: when call simhash_model.display() it seems to always be printed in console and not only the logs
        logging.debug(f'{simhash_model.display()}') # need to be called after optimize()
        simhash_model.printStats() # also exists []https://www.gurobi.com/documentation/9.1/refman/py_model_printstats.html
    # Otherwise could write model to file
    # m.write('file.lp')

    for v in simhash_model.getVars():
        logging.debug('%s %g' % (v.varName, v.x))

    logging.debug('Obj: %g' % simhash_model.objVal)

    # Status checking
    status = simhash_model.Status
    # Cannot happen as 0 is always solution
    # if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
    #     print('The model cannot be solved because it is infeasible or '
    #           'unbounded')
    #      sys.exit(1)

    if status != GRB.OPTIMAL:
        print('Optimization was stopped with status ' + str(status))
        # sys.exit(1)
    if PARAMETERS['find_mult_sol']:
        # In the case where PoolSearchMode = 2
        # OPTIMAL means found specified number of solution or prove there are less
        # https://www.gurobi.com/documentation/9.5/refman/examples.html
        if status == GRB.OPTIMAL:
            logging.info(f'int prog found all sol ?')
            if counter_stats is not None:
                counter_stats['gurobi_ip_found_all_sol'] += 1 # found all sol when ask for multiples



    # Print number of solutions stored
    nSolutions = simhash_model.SolCount
    if counter_stats is not None:
        counter_stats['gurobi_total[include 0]_solcount'] += nSolutions # this include 0 sols
    logging.d2bg('Number of solutions found: ' + str(nSolutions))

    # Print objective values of solutions
    # Also recover solutions as movie histories
    sol_obj_dict = dict()
    histories_subset = []
    for e in range(nSolutions):
        # check https://www.gurobi.com/documentation/9.0/refman/xn.html#attr:Xn
        # Solution are sorted in order of worsening objective value
        simhash_model.setParam(GRB.Param.SolutionNumber, e)
        # Recover objective
        cur_sol_obj = simhash_model.PoolObjVal
        if cur_sol_obj in sol_obj_dict:
            sol_obj_dict[cur_sol_obj] += 1 # increment sol count
        else:
            sol_obj_dict[cur_sol_obj] = 1 # create value for key
        # print('%g ' % simhash_model.PoolObjVal, end='')
        # if e % 15 == 14:
        #     print('')
        if cur_sol_obj > 0:
            selected_history_subset = []
            counter = 0
            for i in range(len(hash_history)):
                if subset_selection[i].Xn == 1:
                    selected_history_subset.append(movie_id_hist[counter])
                # else subset_selection[i] should be 0.
                counter += 1
            histories_subset.append(selected_history_subset)
        elif cur_sol_obj == 0:
            if counter_stats is not None:
                # count number of int prog where enumerated all solutions ?
                # No for that need to check if model.Status says optimal (done after .optimize() above)
                pass
                counter_stats['gurobi_ip_nonzero_sol_count'] += nSolutions-1 # remove the unwanted 0 solution


    # print('')
    logging.d2bg(f'Solution count (values) by objective value (keys): {sol_obj_dict}')

    return simhash_model, subset_selection, histories_subset


if __name__ == '__main__':
    multisol_test = False
    simple_example = True

    if multisol_test:
        # Imports only necessary for main
        from FLoC.preprocessing.movielens_extractor import precompute_cityhash_movielens
        hash_for_movie, title_for_movie, movie_for_title, movie_id_list = precompute_cityhash_movielens()

        # Test and also print out integer program
        SEED = 3
        SimHashBitLen = 20
        vocab_filepath = f'../GAN/save_ml25m/LeakGAN_ml25m_vocab_5000_32.pkl'
        needed_params = {'gurobi_timelimit': 100, 'find_mult_sol': True, 'data_gen_source': 'test', 'timer_nametag': 'test'}

        import pickle
        word_ml25m, vocab_ml25m = pickle.load(open(vocab_filepath, mode='rb'))

        from FLoC.attack.generating_histories import generate_from_traintest  # circular import if imported for whole file
        target_simhashes, target_histories = generate_from_traintest('../GAN/save_ml25m/realtest_ml25m_5000_32.txt', 1,
                                                                     title_for_movie, SimHashBitLen, word_ml25m,
                                                                     seed=SEED)

        print(f'target simhash {target_simhashes}, target history {target_histories}')
        target_hash_history = []
        target_movie_id_history = []
        for movie_title in target_histories[0]:
            target_movie_id_history.append(movie_for_title[movie_title])
            target_hash_history.append(hash_for_movie[movie_for_title[movie_title]])

        target_bits = bin(target_simhashes[0])[2:2 + SimHashBitLen]
        r_target_bits = target_bits[::-1]

        utils.create_logging_levels() # needed as timer func in solve gurobi uses d5bg
        utils.init_loggers(log_to_stdout=True, log_to_file=False, sh_lvl=logging.DEBUG) # call func above if not already done (call this one for debug prints)
        simhash_model, subset_selection, histories_subset = \
            solve_ip_gurobi(target_hash_history, SimHashBitLen, r_target_bits, target_movie_id_history, needed_params,
                            max_sol_count=2000000000, print_ip=True)

    if simple_example:
        # Check chromium_floc/floc_sample.py
        simhash_len = 5
        needed_params = {'gurobi_timelimit': 10, 'find_mult_sol': True, 'data_gen_source': 'test', 'timer_nametag': 'test'}

        domains1 = ['google.com', 'youtube.com', 'facebook.com'] # {'google.com', 'youtube.com', 'facebook.com'}
        domains2 = ['google.com', 'youtube.com', 'netflix.com'] # {'google.com', 'youtube.com', 'netflix.com'}
        from FLoC.chromium_components import cityhash
        def cityhash_domains(domains):
            cityhashed_domains = list() # set()
            for domain in domains1:
                if isinstance(cityhashed_domains, set):
                    cityhashed_domains.add(cityhash.hash64(domain))
                elif isinstance(cityhashed_domains, list):
                    cityhashed_domains.append(cityhash.hash64(domain))
                else:
                    raise Exception('unsuported type')
            return cityhashed_domains

        cityhashed_domains1 = cityhash_domains(domains1)
        cityhash_domains2 = cityhash_domains(domains2)

        target_simhash = '10011'
        r_target_bits = target_simhash[::-1]

        utils.create_logging_levels()  # needed as timer func in solve gurobi uses d5bg
        utils.init_loggers(log_to_stdout=True, log_to_file=False, sh_lvl=logging.DEBUG)  # call func above if not already done (call this one for debug prints)

        simhash_model1, subset_selection1, histories_subset1 = \
            solve_ip_gurobi(cityhashed_domains1, simhash_len, r_target_bits, domains1, needed_params, max_sol_count=2000000000, print_ip=True)

        # Now if want a different simhash
        target_simhash = '10111'
        r_target_bits = target_simhash[::-1]
        simhash_model10, subset_selection10, histories_subset10 = \
            solve_ip_gurobi(cityhashed_domains1, 5, r_target_bits, domains1, needed_params,
                            max_sol_count=2000000000, print_ip=True)

        # verification
        computed_simhash = sim_hash.sim_hash_strings(set(histories_subset10[0]), output_dimensions=5)
        print(f"{bin(computed_simhash)[2:]} == {target_simhash} ? {str(bin(computed_simhash)[2:]) == target_simhash}")


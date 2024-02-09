# basic Adaptive Large Neighborhood Search for MiniZinc
# made by Mateusz Ślayński for the teaching purposes
from minizinc import Instance, Model, Solver
import random
import time
from datetime import timedelta

problem_path = "data/competition.dzn"
init_model_path = "pp_init.mzn"
improve_model_path = "comp-improve.mzn"
# how many variables should be relaxed 
initial_zeroing_rate = 0.07
# how muchdd zeroing rate should increase when there is no progress
adaption_rate = 0.01
# after what should we increase/adapt the zeroing rate
timelimit = 15

zeroing_rate = initial_zeroing_rate

# we use MiniZinc library to get the gecode solver
gecode = Solver.lookup("gecode")


def initial_solution():
    """
    Finds the initial solution
    Returns tuple (route, distance)
    """
    print("searching for initial solution...")
    initial_model = Model()
    initial_model.add_file(init_model_path)
    initial_model.add_file(problem_path)

    initial_instance = Instance(gecode, initial_model)
    res = initial_instance.solve(processes=10, optimisation_level=3)
    return res["solution"], res["obj"]


# we use MiniZinc library to create the improving model for given data instance
improve_model = Model()
improve_model.add_file(improve_model_path)
improve_model.add_file(problem_path)
improve_instance = Instance(gecode, improve_model)


def cfp3(solution, rate):
    seq_len = 5
    seq_n = 2
    sol_len = len(solution)
    fixed_solution = solution.copy()

    zero_positions = [0] * seq_n + [1] * (sol_len - seq_len + 5)
    random.shuffle(zero_positions)

    for i in range(sol_len):
        if i < sol_len - 4 and zero_positions[i] == 0:
            fixed_solution[i] = 0
            fixed_solution[i + 1] = 0
            fixed_solution[i + 2] = 0
            fixed_solution[i + 3] = 0

    print(fixed_solution)

    return fixed_solution


def cfp2(solution, rate):
    seq_len = 4
    seq_n = 4
    fixed_solution = solution.copy()

    sol_len = len(solution)
    # print(f"resetting {seq_n} sequences of {seq_len} variables")
    zero_positions = [0] * seq_n + [1] * (int(sol_len / seq_len) - seq_n + 1)
    random.shuffle(zero_positions)
    zero_positions_zip = zip(zero_positions, zero_positions, zero_positions, zero_positions)
    zero_final = []
    for tpl in zero_positions_zip:
        zero_final.extend(tpl)

    for i in range(sol_len):
        if zero_final[i] == 0:
            fixed_solution[i] = 0

    # print(fixed_solution)

    return fixed_solution


def create_fixed_part(solution, rate):
    """
    This function should relax random {rate}% of the solution.
    Solution is just a list of numbers.
    We just have to create a new list where random {rate}% of the numbers are zeros!
    """
    fixed_solution = solution.copy()
    # TODO:
    # - set random part of the fixed_solution to zeros!

    sol_len = len(solution)
    n = int(rate * sol_len)
    # print(f"resetting {n} variables")
    zero_positions = [0] * n + [1] * (sol_len - n)
    random.shuffle(zero_positions)

    for i in range(sol_len):
        if zero_positions[i] == 0:
            fixed_solution[i] = 0

    # print(fixed_solution)

    return fixed_solution


def improve_solution(solution, rate, old_obj):
    """
    This function improves the given solution.
    """
    # fixed_solution = create_fixed_part(solution, rate)
    fixed_solution = cfp3(solution, rate)
    # the branch method creates a new "copy of the model"
    with improve_instance.branch() as opt:
        # then we set the initial_solution
        opt["initProductionDate"] = fixed_solution

        try:
            res = opt.solve(timeout=timedelta(seconds=10), optimisation_level=1, processes=4)
            return res["productionDate"], res["objective"]
        except Exception as e:
            print(e, 'hihi')
            return solution, 40000


# just to calculate how much time we spent on the optimization
checkpoint = time.time()
# best_solution, best_obj = initial_solution()
# print(best_solution, best_obj, sep="\n")

# best_solution = [2, 5, 10, 8, 11, 4, 7, 6, 12, 1, 3, 9]


best_solution = [16, 116, 139, 150, 40, 111, 109, 110, 112, 140, 141, 142, 59, 58, 60, 72, 144, 143, 22, 54, 53, 82, 83, 81, 84, 85, 126, 125, 127, 149, 21, 56, 55, 75, 76, 77, 73,
                 130, 128, 131, 74, 129, 29, 32, 31, 30, 57, 93, 148, 42, 41, 108, 135, 136, 147, 38, 37, 63, 39, 123, 133, 132, 124, 134, 48, 50, 49, 52, 51, 94, 97, 95, 99, 98,
                 96, 146, 27, 44, 43, 28, 45, 100, 137, 138, 19, 20, 18, 17, 46, 47, 119, 117, 118, 145, 64, 114, 115, 113, 15, 66, 69, 68, 67, 65, 71, 70, 122, 120, 121, 24, 26,
                 23, 25, 61, 62, 78, 80, 91, 88, 79, 89, 90, 92, 87, 86, 9, 13, 34, 35, 33, 14, 36, 101, 102, 107, 106, 104, 105, 103]

best_obj = 100000

while True:
    # we improve the the current solution
    new_solution, new_obj = improve_solution(best_solution, zeroing_rate, best_obj)
    # if it's better than the old one
    print('new obj', new_obj)
    if new_obj < best_obj:
        checkpoint = time.time()
        # we reset the zeroing rate
        zeroing_rate = initial_zeroing_rate
        # we remember the best solution
        best_solution = new_solution
        best_obj = new_obj
        print(f"- cost: {best_obj}")
        print(f"- sol: {best_solution}")
        print("-----------------------")

    # if the solver struggles we increase the zeroing rate :)
    if time.time() - checkpoint > timelimit:
        print(".")
        checkpoint = time.time()
        # zeroing_rate += adaption_rate
        # print(f"* changed zeroing rate to {zeroing_rate}")

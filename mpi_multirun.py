"""
MPI multiple instance execution of ENDOF (Endof New Distributed Optimization
Framework)

This allows to launch an optimization problem on an MPI enabled cluster in
several modes. Two of them are implemented here:
- several independent runs, one per node except node 0 which takes care of the
  control loop and reporting. This version checks that at least two nodes are
  present and exits otherwise.
- several independent runs plus a run that receives the best solutions from the
  rest of instances (based on extesion of statistical ensemble methods to
  evolutive metaheuristics). This version checks that at least four nodes are
  present and exits otherwise.

The optimization methods available for use are those of the alg module.

This file is distributed under the MIT license (http://opensource.org/licenses/MIT)
"""


import os
import sys
import getopt
from mpi4py import MPI
from parsetsp import parsetsp
import alg


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Modes
# Several independent instances, equivalent to Multistart
MULTISTART = 0
# Independent instances plus one that receives the best solutions of the others
MULTIVERSE = 1
# Select mode

# Get execution parametres from command line arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "hm:a:f:r:i:s:")
except getopt.GetoptError:
    print "Error parsing command line"
    sys.exit(2)

mode = None
alg_selection = None
inputfile = None
report_step = 10
maxiter = 100
seed = None


def print_help():
    print("mpi_multirun.py -m mode -a alg -f inputfile -r report_step -i iterations -s seed")
    print("mode: MULTISTART | MULTIVERSE")
    print("alg: ga | aco")


for opt, arg in opts:
    if opt == '-h':
        if rank == 0:
            print("mpi_multirun.py -m mode -a alg")
            print("mode: MULTISTART | MULTIVERSE")
            print("alg: ga | aco")
        sys.exit()
    elif opt == '-m':
        if arg.lower() == "multistart":
            mode = MULTISTART
        elif arg.lower() == "multiverse":
            mode = MULTIVERSE
        else:
            if rank == 0:
                print("Unknown mode:", arg)
            sys.exit(2)
    elif opt == '-a':
        arg_lower = arg.lower()
        if arg_lower in ("ga", "aco"):
            alg_selection = arg_lower
        else:
            if rank == 0:
                print("Unknown mode:", arg)
            sys.exit(2)
    elif opt == '-f':
        inputfile = arg
    elif opt == '-r':
        report_step = int(arg)
    elif opt == '-i':
        maxiter = int(arg)
    elif opt == '-s':
        seed = float(arg)

# Check input
if mode is None or alg_selection is None or inputfile is None:
    if rank == 0:
        print_help()
    sys.exit()

# Check init conditions
if (mode == MULTISTART and size < 2) or (mode == MULTIVERSE and size < 4):
    if rank == 0:
        print("Too few processes", rank, "for  mode", mode)
    exit()

# Algorithms
if alg_selection == 'ga':
    if rank== 0:
        pass
    elif rank >= 1:
        from alg.ga_tsp import ga_tsp
        tsp = parsetsp(inputfile)
        myalg = ga_tsp(tsp.cm, elitism=2, rand_seed=seed, rand_offset=17*rank)
        myalg.initialize_population()
elif alg_selection == 'aco':
    if rank== 0:
        pass
    elif rank >= 1:
        from alg.aco_tsp import aco_tsp
        tsp = parsetsp(inputfile)
        myalg = aco_tsp(tsp.cm, rand_seed=seed, rand_offset=17*rank)
else:
    if rank == 0:
        print("Unrecognized algorithm", alg_selection)
    exit()

# Tag constants
SEND_SOL = 19
SEND_MULTIV_SOL = 17
UPDATE_SOLS = 13

# Parametre init
if mode == MULTISTART:
    num_workers = size - 1
    num_clones = num_workers
    multiverse_process = None
    first_clone = 1
elif mode == MULTIVERSE:
    num_workers = size - 1
    num_clones = num_workers - 1
    multiverse_process = 1
    first_clone = 2
else:
    if rank == 0:
        print("bad mode")
    exit()

# Start up the processes
comm.barrier()

# Run iterations
next_step = True
new_best = sys.float_info.max
best_obj = sys.float_info.max
best_sol = sys.float_info.max

if mode == MULTISTART:
    if rank == 0:
        for i in range(maxiter):
            # signal workers for next iteration
            comm.bcast(next_step, root=0)
            # wait for all workers to perform the step: receive best objs
            new_best_obj, new_best = comm.reduce(best_obj, op=MPI.MINLOC)
            # update minimum cost and the solution that yielded it
            if new_best_obj >= best_obj or new_best == 0:
                comm.bcast(-1, root=0)
            else:
                comm.bcast(new_best, root=0)
                best_sol = comm.recv(source=new_best, tag=SEND_SOL)
                best_obj = new_best_obj
            if (i+1) % report_step == 0:
                print("iteration: {}; best sol: {}".format(i+1, best_obj))
        # Signal workers for exit signal
        next_step = False
        comm.bcast(next_step, root=0)
        # Last update + iteration step for the multiverse worker
    elif rank >= first_clone:
        while True:
            # Wait for next_iteration or exit signal
            next_step = comm.bcast(next_step, root=0)
            if not next_step:
                break
            # Run iteration
            myalg._run_iteration()
            best_obj = myalg._best_obj
            best_sol = myalg._best_sol
            # Send best obj so far to signal completion
            comm.reduce(best_obj, op=MPI.MINLOC)
            new_best = comm.bcast(new_best, root=0)
            if new_best == rank:
                comm.send(best_sol, dest=0, tag=SEND_SOL)
    else:
        print("This is not OK")

elif mode == MULTIVERSE:
    if rank == 0:
        best_sols = []
        for i in range(maxiter - 1):
            # Update multiverse worker with new solutions
            comm.send(best_sols, dest=multiverse_process, tag=UPDATE_SOLS)
            # signal workers for next iteration
            comm.bcast(next_step, root=0)
            # wait for all workers to perform the step: receive best objs
            best_objs = []
            best_sols = []
            for _ in range(num_clones):
                received_obj, received_sol = comm.recv(source=MPI.ANY_SOURCE,
                                                       tag=SEND_SOL)
                best_objs.append(received_obj)
                best_sols.append(received_sol)
            multiv_obj, multiv_sol = comm.recv(source=multiverse_process,
                                               tag=SEND_MULTIV_SOL)
            clones_best_obj, clones_best_sol = min(zip(best_objs, best_sols))
            if multiv_obj <= clones_best_obj:
                new_best_obj = multiv_obj
                new_best_sol = multiv_sol
            else:
                new_best_obj = clones_best_obj
                new_best_sol = clones_best_sol
            # update minimum cost and the solution that yielded it
            if new_best_obj < best_obj:
                best_obj = new_best_obj
                best_sol = new_best_sol
            if (i+1) % report_step == 0:
                print("iteration: {}; best sol: {}".format(i+1, best_obj))
        # Last update + iteration step for the multiverse worker
        comm.send(best_sols, dest=multiverse_process, tag=UPDATE_SOLS)
        # Signal workers for exit signal
        next_step = False
        comm.bcast(next_step, root=0)
        # Multiverse process will still process the last set of solutions
        new_best_obj, new_best_sol = comm.recv(source=multiverse_process,
                                               tag=SEND_MULTIV_SOL)
        # update minimum cost and the solution that yielded it
        if new_best_obj < best_obj:
            best_obj = new_best_obj
            best_sol = new_best_sol
    elif rank >= first_clone:
        while True:
            # Wait for next_iteration or exit signal
            next_step = comm.bcast(next_step, root=0)
            if not next_step:
                break
            # Run iteration
            myalg._run_iteration()
            best_obj = myalg._best_obj
            best_sol = myalg._best_sol
            # Send best obj so far to signal completion
            comm.send((best_obj, best_sol), dest=0, tag=SEND_SOL)
    elif rank == multiverse_process:
        while True:
            # Receive new solutions
            new_sols = comm.recv(source=0, tag=UPDATE_SOLS)
            myalg._incoming_population = new_sols
            # Wait for next_iteration or exit signal
            next_step = comm.bcast(next_step, root=0)
            # Run iteration
            myalg._run_iteration()
            best_obj = myalg._best_obj
            best_sol = myalg._best_sol
            # Send best obj so far to signal completion
            comm.send((best_obj, best_sol), dest=0, tag=SEND_MULTIV_SOL)
            # Check next step after running iteration because the multiverse
            # process performs a last iteration after the stop is signaled
            # to integrate the last solutions from the other processes
            if not next_step:
                break
    else:
        print("This is not OK")

# Report solution
if rank == 0:
    print("iteration: {}; best sol: {}".format(maxiter, best_obj))
    print("solution: {}".format(best_sol))
    if seed is not None:
        print("random seed: {}".format(seed))

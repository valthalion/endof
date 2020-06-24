"""
Run a set of instances to tests ENDOF (Endof New Distributed Optimization
Framework)

This reads the set of problem instances included with ENDOF and launches the MPI
parallel execution for each one a specified number of times, writing the results
to individual log files.

The aim is to automate the running of problem instances in order to generate data
for performance evaluation of the different methods.

This file is distributed under the MIT license (http://opensource.org/licenses/MIT)
"""

import os
import subprocess
import re
import random


tspproblems = os.listdir("tspsamples")
tspproblems.remove('bestknownsols')


def execcommand(folder, problem, num_procs, alg, method, report_step, max_iter, n):
    seed = random.random()
    logfile = "{}_{}_{}_{}_{}.log".format(problem, num_procs, alg, method, n)
    return "time /usr/bin/mpiexec -n {} /usr/bin/python mpi_multirun.py -f {}/{} -a {} -m {} -r {} -i {} -s {} &> {}".format(num_procs, folder, problem, alg, method, report_step, max_iter, seed, logfile)


def iters_from_name(name, mult=10):
    m = re.search('''\d{2,}''', name)
    num = int(m.group(0))
    return num * mult

# TODO: Update number of processes and instances to launch in RPi cluster
num_procs = 4
instances = 25
report_step = 1

for problem in tspproblems:
    max_iter = iters_from_name(problem, 10)

    for s in range(instances):
        cmd = execcommand("tspsamples", problem, num_procs, 'ga', 'multistart', report_step, max_iter, s)
        subprocess.call(cmd, shell=True)
        cmd = execcommand("tspsamples", problem, num_procs, 'aco', 'multistart', report_step, max_iter, s)
        subprocess.call(cmd, shell=True)
        cmd = execcommand("tspsamples", problem, num_procs, 'ga', 'multiverse', report_step, max_iter, s)
        subprocess.call(cmd, shell=True)
        cmd = execcommand("tspsamples", problem, num_procs, 'aco', 'multiverse', report_step, max_iter, s)
        subprocess.call(cmd, shell=True)

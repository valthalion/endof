# ENDOF: Endof New Distributed Optimization Framework 

## Introduction

This repository contains the software used to test the Multiverse method, a generic procedure for any population-based metaheuristic that improves on the basic multistart approach, while incurring negligible additional overhead in terms of running time, communications, and complexity for the developer. More efficient methods exist in the state of the art, but they require more computation capacity and more complexity in the development. This paper (link to come shortly) provides the full detail.


## Repository structure

The root contains the scripts needed to run the tests described in the paper and analyze the results.

The `alg` folder contains the base Genetic Algorithm (GA) and Ant Colony Optimization(ACO) algorithms adapted to run distributed across an MPI system, such as a cluster, as well as specialized versions for solving the Travelling Salesman Problem (TSP).

The `tspsamples` folder contains the asymmetric TSP instances provided in the benchmark library [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), which are used for the tests. They are provided here for convenience, as the scripts will retrieve them from this location.

The `data` folder contains a zip file with the output of the tests run in the paper mentioned above. The scripts for analysis will work if the archive is uncompressed directly in the folder, producing the results reported.

The `endofdb.sql` script run on a MySQL database will generate the schema and user that the analysis expects.

The scripts work as follows.


### `mpi_multirun.py`

This runs one of the TSP instances several times using either GA or ACO in Multistart or Multiverse mode. It requires `mpi4py` and can be called as:

    ```
    python mpi_multirun.py -m <mode> -a <alg> -f <inputfile> -r <report_step> -i <iterations> -s <seed>
    ```
The `<mode>` can be `MULTISTART` or `MULTIVERSE`, `<alg>` can be `ga` or `aco`, `<inputfile>` is the path to a file describing the TSP problem in the same format as TSPLIB, `report_step` is an int specifying the number of interations of the algorithm between updates in the log, `<iterations>` is the number of iterations at which to stop the algorithm, and optionally a `<seed>` for the random number generator can be provided (if not provided, one is randomly generated; in both cases the seed is recorded in the output for reproducibility).

The output is printed to `stdout`. The files in `data` show what information is included in the output.

This uses `parsetsp` to process input files.


### `run_tests.py`

This runs all the input files in `tspsamples`, each using GA and ACO in both Multistart and Multiverse modes, for several repetitions. The repetitions are needed to properly analyze the results of the algorithms as they are stochastic in nature.

The parameters `num_procs` (the number of MPI processes to use), `instances` (the number of repetitions), and `report_step` (as above) can be configured by editing this file. The output of each run is redirected to a file in data with a filename reflecting the TSP instance, algorithm, mode, and repetition.


### `results2db.py`

This script takes the output files in `data` and processes them to populated a database, as specified by `endofdb.sql`. It relies on the location of the data and the naming convention used by `run_tests.py`.


### `analyzebd.py`

This script makes the analysis described in the paper starting from the data in the database.

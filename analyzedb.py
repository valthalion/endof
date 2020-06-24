# -*- coding: utf-8 -*-
"""
Analyze the results of running ENDOF by performing several calcualtions and
plots on the information stored in the endof database.

For each type of experiment (ga, aco, sched), build:
- a table of the percentage improvement of multiverse in average and best
  solutions compared to the multistart results, and percentage improvement
  in running and wall time
- a boxplot of the (normalized) results with either method for each experiment
  in terms of best solution achieved and anytime performance measured as
  hypervolume indicator
- tests to assess statistical differences in performance (same as the boxplots)

Copyright Diego Diaz Fidalgo 2015/06/18

This file is distributed under the MIT license (http://opensource.org/licenses/MIT)
"""

import MySQLdb
import pylab
import math
import numpy as np
from scipy.stats import mannwhitneyu as mwutest


infinity = float('inf')


def print_table(table, cols, latex=False):
    """
    Print a table with the data for each instance

    The table contains the following fields:
    - Instance name
    - Percentage improvement in average solution
    - Percentage improvement in best solution
    - Percentage increase in average running time
    - Percentage increase in average running time
    - Percentage improvement in hypervolume[*]

    A negative value in the improvement percentages indicates that the
    multiverse solution, time or hypervolume is better (lower) than the
    multistart.

    If latex=True is specified, return it ready to paste into a LaTeX
    document.

    [*] Hypervolume is calculated as area below the best solution-iteration
    curve.
    """
    def n2l(n):
        """
        Convert (float) number n to a string for use in LaTeX table

        The conversion is n -> 'I(n) & D(n)', where I(n) is the integral
        part of n and D(n) is the decimal part of n.
        """
        return '{:+.3}'.format(n).replace('.', ' & ')

    def header_specs(col):
        """
        Generate the LaTeX column specification for a column given the
        column type. The col argument is a tuple (col_type, col_name,
        calc_type, factor). Only col_type is used here, to identify
        which specification to generate.

        At the moment, only 'num' is supported as col_type, and it
        returns '''r@{.}l''' which is used to define two colums to form
        an apparent single numeric column aligned at the decimal point
        (intended to generate cell contents with the n2l function).
        """
        col_type = col[0]
        specs = {'num': '''r@{.}l'''}
        return specs[col_type]

    def header_contents(col):
        """
        Generate de LaTeX column header content for a column given the
        column type. The col argument is a tuple (col_type, col_name,
        calc_type, factor). Only col_type and col_name are used here,
        col_type to identify which specification to generate, and col_name
        to include (verbatim) as content.

        At the moment, only 'num' is supported as col_type, and it
        returns the string that places col_name centered as the header
        of a 2-column merge.
        """
        col_type, col_name = col[0], col[1]
        contents = {'num': ('''\multicolumn{2}{|c|}{''', '''}''')}
        return col_name.join(contents[col_type])

    def get_val(instance, col):
        """
        Calculate the value to use as content in a cell, given the
        instance (row) and column. The col argument is a tuple (col_type,
        col_name, calc_type, factor). Only calc_type (to determine how to
        calculate the value) and factor (to select which fields to use) are
        used.

        Only 'delta' is supported at the moment as calc_type, and it produces
        the percentage variation of the factor field for multiverse with
        respect to the the factor field for multistart.
        """
        col_calc, col_factor = col[2], col[3]
        if col_calc == 'delta':
            val_mv = instance[col_factor + '_mv']
            val_ms = instance[col_factor + '_ms']
            diff = 100 * (val_mv - val_ms) / val_ms
            return diff
        
    # Print header
    if latex:
        headers = '|'.join(header_specs(col) for col in cols)
        titles = ' & '.join(header_contents(col) for col in cols)
        print(r'\begin{tabular}{|l||' + headers + r'|}')
        print(r'\hline')
        print(r'Instance & ' + titles + r' \\ \hline \hline')
    else:
        titles = '|'.join('{:>8}'.format(col[1]) for col in cols)
        sep = "+----------------+" + '+'.join('--------' for _ in cols)
        print("|Instance        |" + titles + "|")
        print(sep)

    # Print body
    for instance in table.values():
        name = instance['name']
        vals = (get_val(instance, col) for col in cols)
        if latex:
            print(name + r' & ' + ' & '.join(n2l(val) for val in vals) +
                  r' \\ \hline')
        else:
            print('|{:<15} | '.format(name) +
                  ' | '.join('{:>+6.3}'.format(val) for val in vals) + ' |')

    # Print closing
    if latex:
        print(r'\end{tabular}')
    else:
        print(sep)
    print()


def draw_boxplot(table, factor, exp_type):
    """
    Generate a boxplot from the data in table for the factor provided, and
    evaluate the Mann-Whitney U test for statistical significance

    draw_boxplot(table, factor, exp_type)

    Inputs:
    - table: a dictionary with the data, should contain at least the
             following fields:
             * 'name': the instance name
             * '<factor>s_mv', and '<factor>s_ms': the vectors containing
               the values for the multiverse and multistart experiments,
               respectively
            * '<factor>_avg_ms', and '<factor>_min_ms': average and minimum
              values in the multistart experiments
    - factor: the factor to use, such as 'best_sol' or 'hypervol'; it is used
              to construct the fields used for table as shown above
    - exp_type: the type of experiment ('ga' or 'aco'), used for
                labelling the graph

    Outputs:
    - Shows a boxplot graph with each MV and corresponding MS experiment adjacent
      for comparison. To ensure that the graph is useful, each pair is normalized
      by subtracting the minimium MS value and dividing by the average MS value.
      This makes all pairs similar in size, while keeping the relative sizes of
      the boxes within each pair.
    - Prints a (rough) table showing the results of the Mann-Whitney U test
      performed on each pair to assess if there is a statistical difference
      between the MV and MS versions of the instance. Since both datasets are
      normalized in the same (affine) way, the result is not modified. If the
      test indicates a difference (marked as True), the MV version is better
      than the MS version.
    """
    # Initialize boxplot data
    bp_data = []
    bp_labels = []
    
    for instance in table.values():
        avg_ms = instance[factor + '_avg_ms']
        min_ms = instance[factor + '_min_ms']
        norm_factor = avg_ms - min_ms
        # TODO: Find a better condition for outlier removal
        # other than by instance name
        if norm_factor == 0 or instance['name'] == 'br17.atsp':
            continue
        data_mv = [(x - avg_ms) / norm_factor for x in instance[factor + 's_mv']]
        data_ms = [(x - avg_ms) / norm_factor for x in instance[factor + 's_ms']]
        bp_data.append(data_mv)
        bp_data.append(data_ms)
        bp_labels.append(instance['name'] + '.mv')
        bp_labels.append(instance['name'] + '.ms')
        
        # Print Mann-Whitney U test results
        u, p = mwutest(data_mv, data_ms)
        median_mv, median_ms = median(data_mv), median(data_ms)
        # Use 2 * 0.05 for comparison with p: directed test
        print((p <= 0.1), 100 * (median_mv - median_ms) / median_ms,
              instance['name'], u, p, median_mv, median_ms)

    # Draw boxplot
    pylab.figure(figsize=(7, 4), dpi=300)
    pylab.boxplot(bp_data)
    pylab.xticks(range(1, len(bp_labels)+1), bp_labels, rotation=90)
    pylab.xlabel("Experiments")
    pylab.ylabel("Normalized {} value".format(factor))
    pylab.title("Comparison of {} for {}".format(factor, exp_type))
    pylab.show()


def median(x):
    """m = median(x): Calculate the median value of x"""
    # The implementation is awfully inefficient, but good enough for
    # the sizes that will be used
    n = len(x)
    m = n // 2
    y = sorted(x)
    if n % 2 == 0:
        res = (y[m] + y[m-1]) / 2
    else:
        res = y[m]
    return res


def run_query(q):
    """
    rows = run_query(q): Execute the query q and return the result as a list
    of tuples

    The database connection established at the beginning of this module is
    used for running the query
    """
    db.query(q)
    r = db.store_result()
    rows = r.fetch_row(maxrows=0)
    return rows


def initialize_table(table, exp_type, db):
    """
    Take an empty dictionary and build an instance dictionary for each
    instance in the database for the given experiment type exp_type

    initialize_table(table, exp_type, db)
    """
#    rows = run_query("""select distinct instance_id, best_known_sol from """
#                     """{}_instances;""".format(exp_type))
    rows = run_query("""select distinct instance_id from """
                     """{}_instances;""".format(exp_type))
    for row in rows:
        instance = row[0]
        table[instance] = {}
        table[instance]['name'] = instance
        table[instance]['num_runs_mv'] = 0
        table[instance]['num_runs_ms'] = 0
        table[instance]['num_iters_mv'] = infinity
        table[instance]['num_iters_ms'] = infinity
        table[instance]['best_sols_mv'] = []
        table[instance]['best_sols_ms'] = []
        table[instance]['rtimes_mv'] = []
        table[instance]['rtimes_ms'] = []
        table[instance]['wtimes_mv'] = []
        table[instance]['wtimes_ms'] = []
        table[instance]['hypervols_mv'] = []
        table[instance]['hypervols_ms'] = []


def fill_instance_data(table, exp_type, db):
    """
    Read instance data from the database and fill it in in the table

    fill_instance_data(table, exp_type, db)

    Inputs:
    - table: a dictionary initialized by initialize_table()
    - exp_type: the experiment prefix, used to select the right table to read
    - db: the database connection on which to execute the queries
    """
    rows = run_query("""select instance_id, is_multiverse, best_sol_end, """
                     """runtime_user+runtime_system, runtime_wall, run_num """
                     """from {}_instances;""".format(exp_type))
    for row in rows:
        instance = row[0]
        is_multiverse = (int(row[1]) != 0)
        sol = float(row[2])
        rtime = float(row[3])
        wtime = float(row[4])
        num_runs = int(row[5])
        sols_list = 'best_sols_mv' if is_multiverse else 'best_sols_ms'
        rtimes_list = 'rtimes_mv' if is_multiverse else 'rtimes_ms'
        wtimes_list = 'wtimes_mv' if is_multiverse else 'wtimes_ms'
        num_runs_field = 'num_runs_mv' if is_multiverse else 'num_runs_ms'
        table[instance][sols_list].append(sol)
        table[instance][rtimes_list].append(rtime)
        table[instance][wtimes_list].append(wtime)
        if num_runs > table[instance][num_runs_field]:
            table[instance][num_runs_field] = num_runs


def fill_hypervolume_data(table, exp_type, db):
    """
    Calculate hypervolume data from the database and fill it in in the table

    fill_hypervolume_data(table, exp_type, db)

    Inputs:
    - table: a dictionary initialized by initialize_table()
    - exp_type: the experiment prefix, used to select the right table to read
    - db: the database connection on which to execute the queries
    """
    rows = run_query("""select instance_id, is_multiverse, sum(best_sol), """
                     """max(iter_num) from {}_iters group by """
                     """instance_id, is_multiverse, run_num;""".format(exp_type))
    for row in rows:
        instance = row[0]
        is_multiverse = (int(row[1]) != 0)
        hv_list = 'hypervols_mv' if is_multiverse else 'hypervols_ms'
        hypervol = float(row[2])
        table[instance][hv_list].append(hypervol)
        
        num_iters = int(row[3])
        num_iters_field = 'num_iters_mv' if is_multiverse else 'num_iters_ms'
        current_val = table[instance][num_iters_field]
        if num_iters < current_val:
            if current_val != infinity:
                print("Warning: Abnormal iteration number ({}-{})".format(
                    instance, 'MV' if is_multiverse else 'MS'))
            table[instance][num_iters_field] = num_iters


def process_factors(table, factors):
    """
    Calculate MV and MS averages, medians and minima for the specified factors
    """
    for instance in table.values():
        for factor in factors:
            data_mv = instance[factor + 's_mv']
            data_ms = instance[factor + 's_ms']
            factor_avg_mv = sum(data_mv) / len(data_mv)
            factor_avg_ms = sum(data_ms) / len(data_ms)
            factor_median_mv = median(data_mv)
            factor_median_ms = median(data_ms)
            factor_min_mv = min(data_mv)
            factor_min_ms = min(data_ms)
            instance[factor + '_avg_mv'] = factor_avg_mv
            instance[factor + '_avg_ms'] = factor_avg_ms
            instance[factor + '_median_mv'] = factor_median_mv
            instance[factor + '_median_ms'] = factor_median_ms
            instance[factor + '_min_mv'] = factor_min_mv
            instance[factor + '_min_ms'] = factor_min_ms
        

def draw_hv_plots(db, table, exp_type):
    """
    Draw the plots for best solution evolution (hypervolume)
    """
    for instance in table:
        q = """select run_num, iter_num, best_sol from {}_iters """ \
            """where instance_id = '{}' and is_multiverse = {}"""
        
        rows = run_query(q.format(exp_type, instance, 1))
        mv_data = np.empty((table[instance]['num_iters_mv'],
                            table[instance]['num_runs_mv']))
        for row in rows:
            # values in DB are 1-based, convert to 0-based for matrix indexing
            run_num = int(row[0]) - 1
            iter_num = int(row[1]) - 1
            best_sol = float(row[2])
            mv_data[iter_num, run_num] = best_sol

        rows = run_query(q.format(exp_type, instance, 0))
        ms_data = np.empty((table[instance]['num_iters_ms'],
                            table[instance]['num_runs_ms']))
        for row in rows:
            # values in DB are 1-based, convert to 0-based for matrix indexing
            run_num = int(row[0]) - 1
            iter_num = int(row[1]) - 1
            best_sol = float(row[2])
            ms_data[iter_num, run_num] = best_sol

        pylab.figure(figsize=(7, 5), dpi=300)
        pylab.plot(np.mean(ms_data, axis=1), 'r', linewidth=2, label='multistart')
        pylab.plot(np.mean(mv_data, axis=1), 'g', linewidth=2, label='multiverse')
        pylab.xlabel("Iterations")
        pylab.ylabel("Best solution (averaged across runs)")
        pylab.title("Best Solution Evolution - {} - {}".format(instance, exp_type))
        pylab.legend()
        pylab.show()


# Database initialization
db = MySQLdb.connect(host="localhost", user="endof", passwd="endof", db="endof")
cur = db.cursor()

experiment_types = ['ga', 'aco']

for exp_type in experiment_types:
    print()
    print("Running for", exp_type)

    # Gather the data
    table = {}
    initialize_table(table, exp_type,  db)
    fill_instance_data(table, exp_type, db)
    fill_hypervolume_data(table, exp_type, db)
    factors = ['best_sol', 'rtime', 'wtime', 'hypervol']
    process_factors(table, factors)

    # Show results
    cols = [('num', '$\Delta$ Avg', 'delta', 'best_sol_avg'),
            ('num', '$\Delta$ Min', 'delta', 'best_sol_min'),
            ('num', 'RTime', 'delta', 'rtime_avg'),
            ('num', 'WTime', 'delta', 'wtime_avg'),
            ('num', '$\Delta$ HV', 'delta', 'hypervol_avg')]
    print_table(table, cols, latex=True)
    print()
    draw_boxplot(table, 'best_sol', exp_type)
    print()
    draw_boxplot(table, 'hypervol', exp_type)
    draw_hv_plots(db, table, exp_type)

# Database cleanup
cur.close()
db.close()

#!/usr/bin/env python3
#===============================================================================
# finrich.py
#===============================================================================

# Imports ======================================================================

import json

from argparse import ArgumentParser
from functools import partial
from math import log
from multiprocessing import Pool
from pybedtools import BedTool
from random import sample
from scipy.stats import gamma
from statistics import mean, median, variance




# Functions ====================================================================j

def ppa_in_interval(interval, finemap):
    """Compute the total posterior probability mass contained in a
    genomic interval

    Parameters
    ----------
    interval
        a BedTool representing the genomic interval
    finemap
        a BedTool representing the fine mapping data
    
    Returns
    -------
    float
        the total posterior probability mass contained in the interval
    """

    return sum(
        float(i.fields[-1]) for i in finemap.intersect(BedTool([interval]))
    )


def draw_sample(dummy, population, k):
    """Draw a sample total PPA value from the background distribution

    Parameters
    ----------
    dummy
        a dummy variable, used for multiprocessing
    population
        iterable containing the population of values
    k
        size of the sample to draw
    
    Returns
    -------
    float
        the total PPA of the sample drawn
    """

    return sum(sample(population, k))


def permutation_test(
    finemap,
    regions,
    background,
    conf: float = 0.95,
    parametric: bool = True,
    permutations: int = 100_000,
    processes: int = 1
):
    """Perform a permutation test for enrichment of fine-mapping signals in
    a set of genomic regions

    Parameters
    ----------
    finemap
        a BedTool representing the fine mapping data
    regions
        a BedTool representing the genomic regions
    background
        a bedTool representing the background
    conf : float
        confidence level for interval estimates
    parametric : bool
        if true, use parametric esimates of logOR and confidence intervals
    permutations : int
        the number of permutations to perform
    processes : int
        the number of processes to use
    
    Returns
    -------
    dict
        a dictionary with keys 'pval', 'logOR', 'conf_lower', 'conf_upper'
    """

    with Pool(processes=processes) as pool:
        ppa_vals = tuple(
            pool.map(
                partial(ppa_in_interval, finemap=finemap),
                background.intersect(finemap, u=True)
            )
        )
    population =  (
        tuple(sorted(ppa_vals, reverse=True))
        + (0,) * (len(background) - len(ppa_vals))
    )
    max_val = sum(population[:len(regions)])
    observed_val = sum(float(i.fields[-1]) for i in finemap.intersect(regions))

    def log_odds(val):
        if val == 0 or observed_val == max_val:
            return float('inf')
        if val == max_val or observed_val == 0:
            return float('-inf')
        return (
            log(observed_val)
            + log(max_val - val)
            - log(max_val - observed_val)
            - log(val)
        )

    with Pool(processes=processes) as pool:
        empirical_dist = sorted(
            pool.map(
                partial(draw_sample, population=population, k=len(regions)),
                range(permutations)
            )
        )
    pval = sum(val >= observed_val for val in empirical_dist) / permutations
    if parametric:
        empirical_mean = mean(empirical_dist)
        empirical_var = variance(empirical_dist)
        a = empirical_mean ** 2 / empirical_var
        scale = empirical_var / empirical_mean
        mean_pp = gamma.cdf(empirical_mean, a, scale=scale)
        if mean_pp <= conf / 2:
            empirical_conf_lower = 0
            empirical_conf_upper = gamma.ppf(conf, a, scale=scale)
        elif mean_pp >= 1 - conf / 2:
            empirical_conf_lower = gamma.ppf((1 - conf) / 2, a, scale=scale)
            empirical_conf_upper = gamma.ppf(1 - (1 - conf) / 2, a, scale=scale)
        else:
            empirical_conf_lower = gamma.ppf(mean_pp - conf / 2, a, scale=scale)
            empirical_conf_upper = gamma.ppf(mean_pp + conf / 2, a, scale=scale)
        return {
            'pval': pval,
            'logOR': log_odds(empirical_mean),
            'conf_lower': log_odds(empirical_conf_upper),
            'conf_upper': log_odds(empirical_conf_lower)
        }
    else:
        return {
            'pval': pval,
            'logOR': log_odds(median(empirical_dist)),
            'conf_lower': log_odds(
                empirical_dist[int(permutations * 0.95)]
            ),
            'conf_upper': log_odds(
                empirical_dist[int(permutations * 0.05)]
            )
        }


def parse_arguments():
    parser = ArgumentParser(
        description='enrichment of fine mapping probability'
    )
    parser.add_argument(
        'finemap',
        metavar='<path/to/finemap.bed>',
        type=BedTool,
        help='bed file with fine-mapping data'
    )
    parser.add_argument(
        'regions',
        metavar='<path/to/regions.bed>',
        type=BedTool,
        help='bed file with test regions data'
    )
    parser.add_argument(
        'background',
        metavar='<path/to/background.bed>',
        type=BedTool,
        help='bed file with background regions data'
    )
    parser.add_argument(
        '--non-parametric',
        action='store_true',
        help='return non-parametric estimates'
    )
    parser.add_argument(
        '--conf',
        metavar='<float>',
        type=float,
        default=0.95,
        help='confidence level for interval estimates [0.95]'
    )
    parser.add_argument(
        '--permutations',
        metavar='<int>',
        type=int,
        default=10_000,
        help='number of permutations [10000]'
    )
    parser.add_argument(
        '--processes',
        metavar='<int>',
        type=int,
        default=1,
        help='number of processes to use [1]'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    result = permutation_test(
        args.finemap,
        args.regions,
        args.background,
        conf = args.conf,
        parametric = not args.non_parametric,
        permutations=args.permutations,
        processes=args.processes
    )
    print(json.dumps(result))

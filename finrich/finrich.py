#!/usr/bin/env python3
#===============================================================================
# finrich.py
#===============================================================================

# Imports ======================================================================

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pybedtools import BedTool
from random import sample




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
    permutations : int
        the number of permutations to perform
    processes : int
        the number of processes to use
    
    Returns
    -------
    float
        the p-value of the test
    """

    with Pool(processes=processes) as pool:
        ppa_vals = tuple(
            pool.map(
                partial(ppa_in_interval, finemap=finemap),
                background.intersect(finemap, u=True)
            )
        )
    test_val = sum(float(i.fields[-1]) for i in finemap.intersect(regions))
    population = ppa_vals + (0,) * (len(background) - len(ppa_vals))
    with Pool(processes=processes) as pool:
        empirical_dist = pool.map(
            partial(draw_sample, population=population, k=len(regions)),
            range(permutations)
        )
    return sum(val >= test_val for val in empirical_dist) / permutations


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
        '--permutations',
        metavar='<int>',
        type=int,
        default=10_000,
        help='number of permutations'
    )
    parser.add_argument(
        '--processes',
        metavar='<int>',
        type=int,
        default=1,
        help='number of processes to use'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    pval = permutation_test(
        args.finemap,
        args.regions,
        args.background,
        permutations=args.permutations,
        processes=args.processes
    )
    print(pval)

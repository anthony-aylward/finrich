#!/usr/bin/env python3
#===============================================================================
# finrich.py
#===============================================================================

"""Test sites for enrichment with gwas fine-mapping signal"""




# Imports ======================================================================

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pybedtools import BedTool
from random import sample




# Functions ====================================================================j

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
        'sites',
        metavar='<path/to/sites.bed>',
        type=BedTool,
        help='bed file with test sites data'
    )
    parser.add_argument(
        'background',
        metavar='<path/to/background.bed>',
        type=BedTool,
        help='bed file with background sites data'
    )
    parser.add_argument(
        '--permutations',
        metavar='<int>',
        type=int,
        default=100_000,
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


def ppa_in_interval(interval, finemap):
    return sum(
        float(i.fields[-1]) for i in finemap.intersect(BedTool([interval]))
    )


def draw_sample(dummy_n, population, k):
    return sum(sample(population, k))


def permutation_test(
    finemap,
    test_sites,
    background,
    permutations: int = 100_000,
    processes: int = 1
):
    with Pool(processes=processes) as pool:
        ppa_vals = tuple(
            pool.map(
                partial(ppa_in_interval, finemap=finemap),
                background.intersect(finemap, u=True)
            )
        )
    test_val = sum(float(i.fields[-1]) for i in finemap.intersect(test_sites))
    population = ppa_vals + (0,) * (len(background) - len(ppa_vals))
    with Pool(processes=processes) as pool:
        empirical_dist = pool.map(
            partial(draw_sample, population=population, k=len(test_sites)),
            range(permutations)
        )
    return sum(val >= test_val for val in empirical_dist) / permutations


def main():
    args = parse_arguments()
    pval = permutation_test(
        args.finemap,
        args.sites,
        args.background,
        permutations=args.permutations,
        processes=args.processes
    )
    print(f'pval: {pval}')

if __name__ == '__main__':
    main()

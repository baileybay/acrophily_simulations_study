"""
This script runs the desired simulation and data subset
using command line arguments.
"""

from argparse import ArgumentParser
from acrophily_sims.sims import AcrophilySim, MeanAbsDiffSim, ProbDiffSim


# Parse arguments and pass to arguments object:
def get_args(args=None):

    # Initialize argument parser for command line:
    parser = ArgumentParser(prog='Acrophily Simulations')

    # Create command for simulation type:
    parser.add_argument('-s', '--sim_type', default="acrophily",
                        help="Type of simulation you wish to run (acrophily/prob_diff/mean_abs_diff)")

    # Create command for political affiliation:
    parser.add_argument('-p', '--poli_affil', default="left",
                        help="Political affiliation subset you wish to run simulation on (left/right)")

    # Create commands for using fractions of data:
    parser.add_argument('-f', '--frac_data', default=False,
                        help="Whether you wish to run simulation on fraction of data (True/False)", type=bool)
                        
    parser.add_argument("-fs", "--frac_start", default=0.0,
                        help="Fraction of users you wish to start from (beginning at 0.0)", type=float)
    parser.add_argument("-fe", "--frac_end", default=1.0,
                        help="Fraction of users you wish to end at (ending at 1.0)", type=float)
                        
    return parser.parse_args(args=args)
    

# Run simulation based on command line arguments:
def run_sim(args):

    if args.sim_type == 'mean_abs_diff':
        sim = MeanAbsDiffSim(poli_affil=args.poli_affil,
        frac_data=args.frac_data,
        frac_start=args.frac_start, frac_end=args.frac_end)
        sim.main()

    elif args.sim_type == 'prob_diff':
        sim = ProbDiffSim(poli_affil=args.poli_affil,
        frac_data=args.frac_data,
        frac_start=args.frac_start, frac_end=args.frac_end)
        sim.main()

    else:
        sim = AcrophilySim(poli_affil=args.poli_affil,
        frac_data=args.frac_data,
        frac_start=args.frac_start, frac_end=args.frac_end)
        sim.main()


# Main function to parse arguments and run simulation:
def main():

    args = get_args()
    run_sim(args)

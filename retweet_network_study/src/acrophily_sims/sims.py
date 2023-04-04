"""
This script contains all functions and classes related to running
any of the three simulations. The three classes are AcrophilySim,
MeanAbsDiffSim, and ProbDiffSim, which all inherit data generated
within the sims_data_prep module's TwitterDataProcessor class.

The AcrophilySim class performs functions that return the probabilities of peer ratings
being more extreme than the ego that retweeted them in the homophily,
acrophily, and empirical conditions within multiple minimum retweet thresholds
(see documentation within AcrophilySim for further details). The ProbDiffSim class
performs functions that return the difference in probability of a peer being more extreme
between the homophily and empirical conditions at the individual level
(see documentation within ProbDiffSim). The MeanAbsDiffSim class performs functions that return
the empirical average mean absolute difference between peers and egos as well as a random baseline
average mean absolute difference where the peer ratings were randomly permuted within
multiple minimum retweet thresholds (see MeanAbsDIffSim).
"""

# Importing libraries
from acrophily_sims.data_prep import TwitterDataProcessor
import numpy as np
import os
import pandas as pd
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportion_confint
import sys

pd.options.mode.chained_assignment = None

# Defining general functions used across multiple simulations:


# Function to print condition statements at beginning of simulation:
def print_condition_statements(poli_affil, frac_data=False, frac_start=None, frac_end=None, sim_type='acrophily'):

    # Print statement for prob diff sim:
    if sim_type == 'prob_diff':
        sim_type_statement = 'probability difference'
    elif sim_type == 'mean_abs_diff':
        sim_type_statement = 'mean absolute difference'
    else:
        sim_type_statement = 'acrophily'

    # Prints statement for specified political affiliation of users:
    if poli_affil == 'right':
        print(f'Beginning {sim_type_statement} simulation for conservative users.', flush=True)
    elif poli_affil == 'left':
        print(f'Beginning {sim_type_statement} simulation for liberal users.', flush=True)

    # Will print start and end fractions if using only a fraction of data:
    if frac_data:
        print(f'Fractions chosen: {frac_start * 100: .1f}% to {frac_end * 100: .1f}%.', flush=True)


# Function to get random 70% subset of users:
def get_random_users(df, fraction=0.7):

    # Get list of unique user IDs:
    users = df['userid'].unique()

    # Get sample size of 70% subset:
    n_sample_users = int(fraction * len(users))

    # Take random subset of users at specified sample size:
    users_subset = np.random.choice(users, size=n_sample_users, replace=False)

    return df[df['userid'].isin(users_subset)].sample(frac=1)


# Gets closest peer rating to an ego:
def get_homophily_peer(peer_ratings, ego_rating):
    # Find absolute differences between peer ratings and ego rating:
    rating_abs_diffs = np.abs(peer_ratings - ego_rating)

    # Find index of minimum absolute difference:
    min_diff_idx = rating_abs_diffs.argmin()

    # Find closest peer at minimum difference index:
    closest_peer_rating = peer_ratings[min_diff_idx]

    return closest_peer_rating, min_diff_idx


# Progressbar function (borrowed from Harvard's COMPSCI 109B: Advanced Topics in Data Science course):
def progressbar(n_step: int, n_total: int):
    """Prints self-updating progress bar to stdout to track for-loop progress

    There are entire 3rd-party libraries dedicated to custom progress-bars.
    A simple function like this is often more than enough to get the job done.

    :param n_total: total number of expected for-loop iterations
    :type n_total: int
    :param n_step: current iteration number, starting at 0
    :type n_step: int

    example:

        for i in range(n_iterations):
            progressbar(i, n_iterations)

    source:

        This function is a simplified version of code found here:
        https://stackoverflow.com/questions/3160699/python-progress-bar/15860757#15860757
    """
    n_step = n_step + 1
    barlen = 50
    progress = n_step / n_total
    block = int(round(barlen * progress))
    status = ""
    if n_step == n_total:
        status = "Done...\r\n\n"
    text = "\r [{0}] {1}/{2} {3}".format(
        "=" * block + "-" * (barlen - block),
        n_step,
        n_total,
        status,
    )
    sys.stdout.write(text)
    sys.stdout.flush()


class AcrophilySim(TwitterDataProcessor):
    """
    The AcrophilySim runs the main simulation that returns the probability of an ego's peers
    being more extreme in the homophily, acrophily, and empirical conditions. It does so
    using three main functions:

    1. get_acrophily_df returns a dataframe with a 70% subset retrieved using get_random_users,
    performing both the homophily and acrophily simulations and appending the results as columns.

    2. get_sim_df runs the get_acrophily_df simulation 100 times per threshold, appending and
    aggregating the results to find the average ego and peer ratings per condition. The probability
    of an ego's peers being more extreme on average is then calculated and stored in a list,
    as well as confidence intervals for this probability for each condition.

    3. get_agg_sim_df aggregates to the threshold level and finds the average results across
    all users by threshold. The probabilities of an ego's peers being more extreme and the confidence
     intervals are then appended.

    The results are then saved to a CSV file using the save_agg_sim_df function. The function 'main' runs
    all steps to create and save the data.
    """

    # Inherit processed data from data prep class:
    def __init__(self, poli_affil, thresholds=range(1, 11), frac_data=False, frac_start=None,
                 frac_end=None, users_file=os.path.join('data', 'users_ratings.csv'),
                 rt_file=os.path.join('data', 'rt_network.csv')):

        # Initialize data with affiliation and data parameters:
        TwitterDataProcessor.__init__(self, poli_affil=poli_affil, frac_data=frac_data, frac_start=frac_start,
                                      frac_end=frac_end, users_file=users_file, rt_file=rt_file)

        # Define joined dataset using TwitterDataProcessor get_retweet_data function:
        self.rt_df = None
        self.curr_rt_df = pd.DataFrame()

        # Initialize simulation threshold range:
        self.thresholds = thresholds

        # Initialize confidence interval lists:
        self.confints_empi = list()
        self.confints_homoph = list()
        self.confints_acroph_min = list()
        self.confints_acroph_max = list()

        # Initialize list of probabilities of peer being more extreme:
        self.probs_more_extreme_empi = list()
        self.probs_more_extreme_homoph = list()
        self.probs_more_extreme_acroph_min = list()
        self.probs_more_extreme_acroph_max = list()

        # Initializing simulation dataframes:
        self.acrophily_df = pd.DataFrame()
        self.threshold_sim_df = pd.DataFrame()
        self.agg_threshold_df = pd.DataFrame()
        self.sim_df = pd.DataFrame()
        self.agg_sim_df = pd.DataFrame()

    # Static method to get acrophily peer:
    @staticmethod
    def get_acrophily_peer_min(peer_ratings, ego_rating):

        # Find differences between peer ratings and ego rating:
        rating_diffs = peer_ratings - ego_rating

        # Find min difference index with added cost to differences below 0:
        acroph_min_diff_idx = np.where(rating_diffs >= 0, rating_diffs, rating_diffs**2 + 100).argmin()

        # Uses index of closest element to grab element:
        closest_peer_rating = peer_ratings[acroph_min_diff_idx]

        return closest_peer_rating, acroph_min_diff_idx
        
    # Static method to get pure acrophily peer:
    @staticmethod
    def get_acrophily_peer_max(peer_ratings, ego_rating):

        # Find differences between peer ratings and ego rating:
        rating_diffs = peer_ratings - ego_rating

        # Find max difference index:
        acroph_max_diff_idx = rating_diffs.argmax()

        # Uses index of farthest element to get element:
        max_diff_peer_rating = peer_ratings[acroph_max_diff_idx]

        return max_diff_peer_rating, acroph_max_diff_idx

    # Get chosen peers for both homophily and acrophily conditions:
    def get_chosen_peers(self, homoph_peer_ratings, acroph_peer_ratings_min, acroph_peer_ratings_max, ego_ratings):

        # Initialize list of closest peers based on homophily simulation:
        homoph_chosen_peer_ratings = []
        acroph_chosen_peer_ratings_min = []
        acroph_chosen_peer_ratings_max = []

        # For each ego rating in subset:
        for ego_rating in ego_ratings:

            # Get homophily and acrophily peer ratings and indexes within respective ratings lists:
            homoph_chosen_peer_rating, homoph_chosen_peer_idx = get_homophily_peer(homoph_peer_ratings,
                                                                                   ego_rating)
            acroph_chosen_peer_rating_min, acroph_chosen_peer_idx_min = self.get_acrophily_peer_min(acroph_peer_ratings_min,
                                                                                        ego_rating)
            acroph_chosen_peer_rating_max, acroph_chosen_peer_idx_max = self.get_acrophily_peer_max(acroph_peer_ratings_max,
                                                                            ego_rating)

            # Remove selected peers from respective peer pools:
            homoph_peer_ratings = np.delete(homoph_peer_ratings, homoph_chosen_peer_idx)
            acroph_peer_ratings_min = np.delete(acroph_peer_ratings_min, acroph_chosen_peer_idx_min)
            acroph_peer_ratings_max = np.delete(acroph_peer_ratings_max, acroph_chosen_peer_idx_max)

            # Append selected peer to respective closest peers list:
            homoph_chosen_peer_ratings.append(homoph_chosen_peer_rating)
            acroph_chosen_peer_ratings_min.append(acroph_chosen_peer_rating_min)
            acroph_chosen_peer_ratings_max.append(acroph_chosen_peer_rating_max)

        return homoph_chosen_peer_ratings, acroph_chosen_peer_ratings_min, acroph_chosen_peer_ratings_max

    # Run homophily simulation and append homophily peer ratings to new column in dataframe:
    def get_acrophily_df(self):

        # Randomize order of dataset for simulation trial:
        self.acrophily_df = get_random_users(self.curr_rt_df)

        # Get all ego and peer ratings:
        ego_ratings = self.acrophily_df['orig_rating_ego'].values
        peer_ratings = self.curr_rt_df['orig_rating_peer'].values

        # Make copy of peer ratings for homophily and acrophily simulations:
        homoph_peer_ratings = np.copy(peer_ratings)
        acroph_peer_ratings_min = np.copy(peer_ratings)
        acroph_peer_ratings_max = np.copy(peer_ratings)

        # Get chosen peers for each condition:
        homoph_chosen_peer_ratings, acroph_chosen_peer_ratings_min, acroph_chosen_peer_ratings_max = self.get_chosen_peers(homoph_peer_ratings,
                                                                                       acroph_peer_ratings_min,
                                                                                       acroph_peer_ratings_max,
                                                                                       ego_ratings)

        # Create new column for closest peers based on strategy:
        self.acrophily_df['homoph_rating_peer'] = homoph_chosen_peer_ratings
        self.acrophily_df['acroph_rating_peer_min'] = acroph_chosen_peer_ratings_min
        self.acrophily_df['acroph_rating_peer_max'] = acroph_chosen_peer_ratings_max

    # Gets proportion of egos with peers more extreme on average:
    def get_prob_more_extreme(self, peer_ratings):
        n_more_extreme = len(self.agg_threshold_df[peer_ratings > self.agg_threshold_df['orig_rating_ego']])
        n_total = len(self.agg_threshold_df)
        prob_more_extreme = n_more_extreme / n_total

        return prob_more_extreme

    # Gets proportion confidence interval for probability of peer being more extreme:
    def get_proportion_confint(self, peer_rating_col):
        n_more_extreme = len(self.agg_threshold_df[peer_rating_col > self.agg_threshold_df['orig_rating_ego']])
        n_total = len(self.agg_threshold_df)
        confint = proportion_confint(n_more_extreme, n_total)

        return confint

    # Runs the main simulation to get acrophily/homophily peers for n=1000 trials
    def run_sim(self, threshold, n=1000):

        # Initialize threshold level dataframe:
        self.threshold_sim_df = pd.DataFrame()
        print(f'Current threshold: {threshold} of {self.thresholds[-1]}', flush=True)

        # Run 100 iterations within current threshold:
        for i in range(n):
            # Progress bar for terminal output:
            progressbar(i, n)

            # Run homophily and acrophily simulation:
            self.get_acrophily_df()

            # Append current trial's dataframe to threshold dataframe
            self.threshold_sim_df = pd.concat([self.threshold_sim_df, self.acrophily_df], axis=0, ignore_index=True)

    # Get probabilities of peer being more extreme at aggregated level:
    def append_prob_more_extreme_lists(self):

        # Define P(peer more extreme) for each condition:
        prob_more_extreme_empi = self.get_prob_more_extreme(self.agg_threshold_df['orig_rating_peer'])
        prob_more_extreme_homoph = self.get_prob_more_extreme(self.agg_threshold_df['homoph_rating_peer'])
        prob_more_extreme_acroph_min = self.get_prob_more_extreme(self.agg_threshold_df['acroph_rating_peer_min'])
        prob_more_extreme_acroph_max = self.get_prob_more_extreme(self.agg_threshold_df['acroph_rating_peer_max'])

        # Append results to list:
        self.probs_more_extreme_empi.append(prob_more_extreme_empi)
        self.probs_more_extreme_homoph.append(prob_more_extreme_homoph)
        self.probs_more_extreme_acroph_min.append(prob_more_extreme_acroph_min)
        self.probs_more_extreme_acroph_max.append(prob_more_extreme_acroph_max)

    # Append P(peer more extreme) confidence intervals to lists for each condition:
    def append_confint_lists(self):

        # Get confidence intervals for each condition for threshold:
        confint_empi = self.get_proportion_confint(self.agg_threshold_df['orig_rating_peer'])
        confint_homoph = self.get_proportion_confint(self.agg_threshold_df['homoph_rating_peer'])
        confint_acroph_min = self.get_proportion_confint(self.agg_threshold_df['acroph_rating_peer_min'])
        confint_acroph_max = self.get_proportion_confint(self.agg_threshold_df['acroph_rating_peer_max'])

        # Append confidence intervals to list:
        self.confints_empi.append(confint_empi)
        self.confints_homoph.append(confint_homoph)
        self.confints_acroph_min.append(confint_acroph_min)
        self.confints_acroph_max.append(confint_acroph_max)

    # Run simulation across multiple thresholds to create simulation dataframe:
    def get_sim_df(self):

        # Print statement to verify that simulation is beginning for proper group/data fraction:
        print_condition_statements(self.poli_affil, self.frac_data, self.frac_start, self.frac_end)

        # Iterate through minimum retweet thresholds:
        for threshold in self.thresholds:

            # Subset original rt dataframe by minimum retweet subset:
            self.curr_rt_df = self.rt_df[self.rt_df['rt'] == threshold]

            # Run simulation:
            self.run_sim(threshold)

            # Aggregate to take mean results for each user for 1000 iterations at threshold:
            self.agg_threshold_df = self.threshold_sim_df.groupby('userid', as_index=False).agg('mean')

            # Add column indicating current threshold:
            self.agg_threshold_df['threshold'] = self.agg_threshold_df['rt'].astype(int)

            # If using full data, keep track of probabilities and confidence intervals now:
            if not self.frac_data:
                self.append_prob_more_extreme_lists()
                self.append_confint_lists()

            # Concatenate with threshold level sim df:
            self.sim_df = pd.concat([self.sim_df, self.agg_threshold_df], axis=0, ignore_index=True)

    # Append probability of peer being more extreme for each condition:
    def get_prob_more_extreme_cols(self):

        self.agg_sim_df['prob_more_extreme_empi'] = self.probs_more_extreme_empi
        self.agg_sim_df['prob_more_extreme_homoph'] = self.probs_more_extreme_homoph
        self.agg_sim_df['prob_more_extreme_acroph_min'] = self.probs_more_extreme_acroph_min
        self.agg_sim_df['prob_more_extreme_acroph_max'] = self.probs_more_extreme_acroph_max

    # Append confidence intervals per threshold for each condition:
    def get_confint_cols(self):

        self.agg_sim_df['confint_empi'] = self.confints_empi
        self.agg_sim_df['confint_homoph'] = self.confints_homoph
        self.agg_sim_df['confint_acroph_min'] = self.confints_acroph_min
        self.agg_sim_df['confint_acroph_max'] = self.confints_acroph_max

    # Get aggregate sim df across thresholds:
    def get_agg_sim_df(self):

        print('Simulation complete. Taking average results by threshold.', flush=True)

        # Group sim df by threshold and get average results across all users:
        self.agg_sim_df = self.sim_df.groupby('threshold', as_index=False).agg('mean')

        # Get P(peer more extreme) and confint columns:
        self.get_prob_more_extreme_cols()
        self.get_confint_cols()

        # Append political affiliation:
        self.agg_sim_df['poli_affil'] = np.repeat(self.poli_affil, len(self.agg_sim_df))
        self.agg_sim_df = self.agg_sim_df[['threshold', 'prob_more_extreme_empi', 'prob_more_extreme_homoph',
                                           'prob_more_extreme_acroph_min', 'prob_more_extreme_acroph_max', 'confint_empi',
                                           'confint_homoph', 'confint_acroph_min', 'confint_acroph_max', 'poli_affil']]

    # Create dataframe file path:
    def get_file_path(self):

        # Base file path on political affiliation:
        if self.poli_affil == 'left':
            path_beginning = os.path.join('data', 'acrophily_sim_left')
        elif self.poli_affil == 'right':
            path_beginning = os.path.join('data', 'acrophily_sim_right')
        else:
            raise Exception('Political affiliation must be defined as left or right.')

        # Add fractions to file name if using fraction of data:
        if self.frac_data:
            file_path = path_beginning + \
                        f'_{self.frac_start}_{self.frac_end}_{self.thresholds[0]}_{self.thresholds[-1]}.csv'
        else:
            file_path = f'{path_beginning}_{self.thresholds[0]}_{self.thresholds[-1]}.csv'

        return file_path

    # Save agg sim df to file path:
    def save_results(self):

        if not self.frac_data:
            print('Average results taken. Saving final dataframe.', flush=True)
        else:
            print('Simulation complete. Saving final dataframe.', flush=True)

        # Get file path:
        file_path = self.get_file_path()

        # Only save if file doesn't already exist:
        assert os.path.exists(file_path) is False, 'File already exists. Will not overwrite.'

        # If simulation ran on full data, save aggregated df::
        if not self.frac_data:
            self.agg_sim_df.to_csv(file_path, index=False)
        else:
            # Save sim df to aggregate later if using only fraction of data:
            self.sim_df = self.sim_df[['userid', 'threshold', 'orig_rating_ego', 'orig_rating_peer',
                                       'homoph_rating_peer', 'acroph_rating_peer_min', 'acroph_rating_peer_max']]
            self.sim_df.to_csv(file_path, index=False)

        print('Dataframe saved.', flush=True)

    # Main function to run the simulation, take the aggregated results, and save to file:
    def main(self):

        # Assert that file path doesn't exist before running simulation:
        file_path = self.get_file_path()
        assert os.path.exists(file_path) is False, 'File already exists. Will not overwrite.'

        # If file path doesn't exist, get data and run simulation:
        self.rt_df = self.get_retweet_data()

        self.get_sim_df()

        # Only get aggregated df if using full data:
        if not self.frac_data:
            self.get_agg_sim_df()

        self.save_results()


class MeanAbsDiffSim(TwitterDataProcessor):
    """
        The MeanAbsDiffSim class runs using four main functions:

        1. get_abs_diff_df uses the general function get_random_users to create a random 70% subset of the data.
        A variable is then created in which the empirical peer ratings from this subset are randomly permuted
        to use as a random baseline comparison. The absolute differences between egos and peers are then calculated
        for both the randomly permuted and empirical conditions and stored as two distinct columns
        in the subset dataframe.

        2. run_full_sim iterates over each minimum retweet threshold in the range of thresholds specified,
        and iterates for 100 trials within each threshold. In each iteration, the abs_diff_df is generated
        with a different 70% subset, and the results are concatenated with a threshold dataframe that is
        re-initialized at the start of each new threshold. At the end of the 100 iterations, an aggregate
        threshold df is created by grouping mean absolute differences by ego and taking the average.
        The aggregate threshold df is then concatenated with the sim df, which stores the average results
        across all thresholds. Confidence intervals for each condition are also generated after each threshold during
        this aggregation step.

        3. get_agg_sim_df takes the resulting sim_df and aggregates the result by threshold, calculating
        the average mean absolute difference across all users for the threshold, as well as confidence
        intervals. It also appends the political affiliation of the users in a column.

        The dataframe is then saved as a CSV file using the save_agg_sim_df function. The function 'main'
        runs all steps to save the generated data.
        """

    # Inherit processed data from data prep class:
    def __init__(self, poli_affil, thresholds=range(1, 11),
                 frac_data=False, frac_start=None, frac_end=None,
                 users_file=os.path.join('data', 'users_ratings.csv'),
                 rt_file=os.path.join('data', 'rt_network.csv')):

        # Initialize data with affiliation and data parameters:
        TwitterDataProcessor.__init__(self, poli_affil=poli_affil, frac_data=frac_data, frac_start=frac_start, frac_end=frac_end,
                                      users_file=users_file, rt_file=rt_file)

        # Define joined dataset using TwitterDataProcessor get_retweet_data function:
        self.rt_df = None
        self.curr_rt_df = pd.DataFrame()

        # Initialize simulation threshold range:
        self.thresholds = thresholds

        # Initialize confidence interval lists:
        self.confints_random = list()
        self.confints_empi = list()

        # Initializing simulation dataframes:
        self.abs_diff_df = pd.DataFrame()
        self.threshold_sim_df = pd.DataFrame()
        self.agg_threshold_df = pd.DataFrame()
        self.sim_df = pd.DataFrame()
        self.agg_sim_df = pd.DataFrame()

    # Function to return abs diff in both random and empirical conditions:
    def get_abs_diff_df(self):

        # Get random 70% subset of users:
        self.abs_diff_df = get_random_users(self.curr_rt_df)

        # Get ego and peer ratings:
        ego_ratings = self.abs_diff_df['orig_rating_ego'].values
        peer_ratings_empi = self.abs_diff_df['orig_rating_peer'].values

        # Get all peers from curent rt df:
        curr_rt_peers = self.curr_rt_df['orig_rating_peer'].values

        # Take random sample of 70% of current rt peer ratings:
        peer_ratings_random = np.random.choice(curr_rt_peers, size=len(ego_ratings), replace=False)

        # Append absolute difference between peer and ego ratings in both conditions:
        self.abs_diff_df['abs_diff_empi'] = np.abs(ego_ratings - peer_ratings_empi)
        self.abs_diff_df['abs_diff_random'] = np.abs(ego_ratings - peer_ratings_random)
        
        # Append random peer ratings to column:
        self.abs_diff_df['random_rating_peer'] = peer_ratings_random

    # Run simulation within single threshold and append results to dataframe:
    def run_threshold_sim(self, n=1000):

        # Initialize threshold df:
        self.threshold_sim_df = pd.DataFrame()

        # Iterate for 100 trials:
        for i in range(n):
            progressbar(i, n)

            # Get abs diff df and concatenate with threshold level df:
            self.get_abs_diff_df()
            self.threshold_sim_df = pd.concat([self.threshold_sim_df, self.abs_diff_df], axis=0, ignore_index=True)

    # Gets mean results of single threshold run as dataframe:
    def get_agg_threshold_df(self, threshold):

        # Initialize agg threshold df:
        self.agg_threshold_df = pd.DataFrame()

        # Subset based on minimum retweet threshold:
        self.curr_rt_df = self.rt_df[self.rt_df['rt'] == threshold]

        print(f'Current threshold: {threshold} of {self.thresholds[-1]}', flush=True)

        # Runs simulation for n=1000 trials and appends to threshold level df:
        self.run_threshold_sim()

        # Take the aggregate of the threshold df after 100 trials, grouped by ego:
        self.agg_threshold_df = self.threshold_sim_df.groupby('userid', as_index=False) \
            .agg(orig_rating_ego=('orig_rating_ego', 'mean'),
                orig_rating_peer=('orig_rating_peer', 'mean'),
                random_rating_peer=('random_rating_peer', 'mean'),
                mean_abs_diff_empi=('abs_diff_empi', 'mean'),
                 mean_abs_diff_random=('abs_diff_random', 'mean'),
                 threshold=('rt', 'mean'))

        # Append threshold to aggregate threshold df:
        self.agg_threshold_df['threshold'] = self.agg_threshold_df['threshold'].astype(int)

    # Append confidence intervals for mean abs diff for each condition to list:
    def append_confint_lists(self):

        # Get threshold level confidence intervals for each condition's mean abs diff:
        confint_empi = sms.DescrStatsW(self.agg_threshold_df['mean_abs_diff_empi'].values).tconfint_mean()
        confint_random = sms.DescrStatsW(self.agg_threshold_df['mean_abs_diff_random']).tconfint_mean()

        # Append confidence intervals to list:
        self.confints_empi.append(confint_empi)
        self.confints_random.append(confint_random)

    # Run simulation performing get_abs_diff for 100 iterations per threshold and store results:
    def run_full_sim(self):

        # Print opening statements to begin simulation:
        print_condition_statements(self.poli_affil, self.frac_data, self.frac_start, self.frac_end,
                                   sim_type='mean_abs_diff')

        # Iterate through each threshold in chosen range:
        for threshold in self.thresholds:

            self.get_agg_threshold_df(threshold)

            # Append confidence intervals to lists now if using full data:
            if not self.frac_data:
                self.append_confint_lists()

            # Concatenate aggregate threshold df with sim_df:
            self.sim_df = pd.concat([self.sim_df, self.agg_threshold_df], axis=0, ignore_index=True)

    # Get aggregate sim df across thresholds:
    def get_agg_sim_df(self):

        print('Simulation complete. Taking average results by threshold.', flush=True)

        # Group sim df by threshold and get average results across all users:
        self.agg_sim_df = self.sim_df.groupby('threshold', as_index=False).agg(userid=('userid', pd.Series.mode),
                orig_rating_ego=('orig_rating_ego', 'mean'),
                orig_rating_peer=('orig_rating_peer', 'mean'),
                random_rating_peer=('random_rating_peer', 'mean'),
                mean_abs_diff_empi=('abs_diff_empi', 'mean'),
                 mean_abs_diff_random=('abs_diff_random', 'mean'))

        # Append confidence intervals per threshold for each condition:
        self.agg_sim_df['confint_empi'] = self.confints_empi
        self.agg_sim_df['confint_random'] = self.confints_random

        # Append political affiliation:
        self.agg_sim_df['poli_affil'] = np.repeat(self.poli_affil, len(self.agg_sim_df))

    # Create dataframe file path:
    def get_file_path(self):

        # Base file path on political affiliation:
        if self.poli_affil == 'left':
            path_beginning = os.path.join('data', 'mean_abs_diff_sim_left')
        elif self.poli_affil == 'right':
            path_beginning = os.path.join('data', 'mean_abs_diff_sim_right')
        else:
            raise Exception('Political affiliation must be defined as left or right.')

        # Add fractions to file name if using fraction of data:
        if self.frac_data:
            file_path = path_beginning + \
                        f'_{self.frac_start}_{self.frac_end}_{self.thresholds[0]}_{self.thresholds[-1]}.csv'
        else:
            file_path = f'{path_beginning}_{self.thresholds[0]}_{self.thresholds[-1]}.csv'

        return file_path

    def save_results(self):

        print('Average results taken. Saving final dataframe.', flush=True)

        file_path = self.get_file_path()

        # Save aggregate df if using full data:
        if not self.frac_data:
            self.agg_sim_df.to_csv(file_path, index=False)
        else:
            # Save sim df to aggregate later if using fraction:
            self.sim_df.to_csv(file_path, index=False)
        print('Dataframe saved.', flush=True)

    # Main function to run all steps of simulation:
    def main(self):
        file_path = self.get_file_path()
        assert os.path.exists(file_path) is False, 'File already exists. Will not overwrite.'

        self.rt_df = self.get_retweet_data()
        self.run_full_sim()

        # Only aggregate if using full data:
        if not self.frac_data:
            self.get_agg_sim_df()

        self.save_results()


# Create probability difference simulation class:
class ProbDiffSim(TwitterDataProcessor):
    """
    The ProbDiffSim class runs using four main functions:

    1. get_homophily_df uses the general function (see above) get_random_users to take a random
    70% subset of the data for the simulation trial. It then iterates through each ego rating in
    this 70% subset and matches the ego rating to the closest available peer, sampling without replacement
    from the list of peer ratings in the entire subset. This simulates a strategy wherein egos choose
    to retweet the closest available peer to them and adds a probabilistic element. It records the results
    in a new column of the subset dataframe to later compare with the empirical peer ratings.

    2. run_sim runs 1000 repetitions of the simulation, where the homophily dataframe is created by calling
    get_homophily_df, and the sim_df is created by concatenating the results of the homophily simulation in
    each iteration. When the 100 trials are complete, a vectorized function (count_more_extreme) creates a
    dummy variable indicating whether a peer is more extreme than the ego for both the homophily and empirical
    conditions, respectively. These counts are then appended as two more columns in the simulation dataframe.

    3. The get_prob_diff_df groups the final sim df by ego rating and takes the mean of the respective
    count_more_extreme columns, converting these columns into probabilities that a peer is more extreme
    for each ego in the dataset in each condition, respectively. A new column is then created by subtracting
    the probability of a peer being more extreme in the homophily condition from the probability of a peer being
    more extreme in the empirical condition. This result gets appended as the 'prob_diff' column in the dataset.

    The file then gets saved using save_prob_diff_df. The function 'main' run the full simulation and saves the
    file.
    """

    # Inherit processed data from data prep class:
    def __init__(self, poli_affil, frac_data=False, frac_start=None, frac_end=None,
                 users_file=os.path.join('data', 'users_ratings.csv'), rt_file=os.path.join('data', 'rt_network.csv')):
        TwitterDataProcessor.__init__(self, poli_affil, frac_data, frac_start,
                                      frac_end, users_file=users_file, rt_file=rt_file)

        # Get processed data
        self.rt_df = None

        # Vectorized lambda function to count whether peer rating is greater than ego rating:
        self.count_more_extreme = np.vectorize(lambda x, y: 1 if y > x else 0)

        # Initializing dataframes that will be created during run of simulation:
        self.homophily_df = pd.DataFrame()
        self.sim_df = pd.DataFrame()
        self.prob_diff_df = pd.DataFrame()

    # Static method to get homophily peers:
    @staticmethod
    def get_homophily_peers(homoph_peer_ratings, ego_ratings):
        # Initialize list of closest peers based on homophily simulation:
        chosen_peer_ratings = []

        # For each ego rating in subset:
        for ego_rating in ego_ratings:
            homoph_peer_rating, homoph_peer_idx = get_homophily_peer(homoph_peer_ratings, ego_rating)

            # Remove selected peer from peer pool:
            homoph_peer_ratings = np.delete(homoph_peer_ratings, homoph_peer_idx)

            # Append selected peer to closest peers list:
            chosen_peer_ratings.append(homoph_peer_rating)

        return chosen_peer_ratings

    # Run homophily simulation and append homophily peer ratings to new column in dataframe:
    def get_homophily_df(self):

        # Randomize order of dataset for simulation trial:
        self.homophily_df = get_random_users(self.rt_df)

        # Get all ego and peer ratings:
        ego_ratings = self.homophily_df['orig_rating_ego'].values
        peer_ratings = self.rt_df['orig_rating_peer'].values

        # Copy peer ratings for homophily condition:
        homoph_peer_ratings = np.copy(peer_ratings)

        # Get chosen peers for homophily condition:
        homoph_chosen_peer_ratings = self.get_homophily_peers(homoph_peer_ratings, ego_ratings)

        # Create new column for closest peers based on homophily strategy:
        self.homophily_df['homoph_rating_peer'] = homoph_chosen_peer_ratings

    # Get columns for both conditions that count whether peer is more extreme:
    def get_count_more_extreme_cols(self):
        # Define trial ego ratings:
        ego_ratings = self.homophily_df['orig_rating_ego']

        # Define trial peer ratings for each condition:
        peer_ratings_empi = self.homophily_df['orig_rating_peer']
        peer_ratings_homoph = self.homophily_df['homoph_rating_peer']

        # Create count column indicating whether peer is more extreme for each condition:
        self.homophily_df['is_more_extreme_homoph'] = self.count_more_extreme(ego_ratings, peer_ratings_homoph)
        self.homophily_df['is_more_extreme_empi'] = self.count_more_extreme(ego_ratings, peer_ratings_empi)

    # Runs simulation for n=1000 repetitions and creates dataframe with user id and probability differences:
    def run_sim(self, n=1000):

        # Print statement based on affiliation/fraction conditions:
        print_condition_statements(self.poli_affil, self.frac_data, self.frac_start, self.frac_end,
                                   sim_type='prob_diff')

        # Add progress bar for 1000 repetitions
        for i in range(n):
            # Establish progress bar:
            progressbar(i, n)

            # Get homophily ratings:
            self.get_homophily_df()

            # Get columns for both conditions that count whether peer is more extreme:
            self.get_count_more_extreme_cols()

            # Append results to dataframe:
            self.sim_df = pd.concat([self.sim_df, self.homophily_df], axis=0, ignore_index=True)

    # Function to get prob diff df, which groups sim df by user ID to get mean results:
    def get_prob_diff_df(self):
        # Get mean probability that a peer is more extreme in each condition for each ego across all trials:
        self.prob_diff_df = self.sim_df.groupby('userid', as_index=False).agg(
            prob_more_extreme_homoph=('is_more_extreme_homoph', 'mean'),
            prob_more_extreme_empi=('is_more_extreme_empi', 'mean'))

        # Get probability of peer being more extreme in each condition:
        prob_more_extreme_empi = self.prob_diff_df['prob_more_extreme_empi']
        prob_more_extreme_homoph = self.prob_diff_df['prob_more_extreme_homoph']

        # Crates column of differences between probabilities in the empirical and homophily conditions:
        self.prob_diff_df['prob_diff'] = prob_more_extreme_empi - prob_more_extreme_homoph
        self.prob_diff_df['poli_affil'] = np.repeat(self.poli_affil, len(self.prob_diff_df))

        print('Dataframe created. Saving to csv.', flush=True)

    # Get file path to save:
    def get_file_path(self):
        # Generate file path based on political affiliation:
        if self.poli_affil == 'left':
            path_beginning = os.path.join('data', 'prob_diff_left')
        elif self.poli_affil == 'right':
            path_beginning = os.path.join('data', 'prob_diff_right')
        else:
            raise Exception('Political affiliation must be defined as left or right.')

        # Add fractions to file name if using fraction of data:
        if self.frac_data:
            file_path = path_beginning + f'_{self.frac_start}_{self.frac_end}.csv'
        else:
            file_path = f'{path_beginning}.csv'

        return file_path

    # Function to save dataframe:
    def save_prob_diff_df(self):

        # Get file path:
        file_path = self.get_file_path()

        # Only save if file doesn't already exist:
        assert os.path.exists(file_path) is False, 'File already exists. Will not overwrite.'

        # Save to CSV:
        self.prob_diff_df.to_csv(file_path, index=False)
        print('Dataframe saved.', flush=True)

    # Main function to run all steps of simulation:
    def main(self):

        # Assure file path doesn't exist before beginning simulation:
        file_path = self.get_file_path()
        assert os.path.exists(file_path) is False, 'File already exists. Will not overwrite.'

        # If file path exists, get retweet data and run sim/save results:
        self.rt_df = self.get_retweet_data()
        self.run_sim()
        self.get_prob_diff_df()
        self.save_prob_diff_df()

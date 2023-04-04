from .sims_data_prep import TwitterDataProcessor
import os
import numpy as np
import time
import pandas as pd

import statsmodels.stats.api as sms

pd.options.mode.chained_assignment = None


# General function shared amongst simulation classes to get random 70% subset of users:
def get_random_users(df, fraction=0.7):
    # Get list of unique user IDs:
    users = df['userid'].unique()

    # Get sample size of 70% subset:
    n_sample_users = int(fraction*len(users))

    # Take random subset of users at specified sample size:
    users_subset = np.random.choice(users, size=n_sample_users, replace=False)

    return df[df['userid'].isin(users_subset)]


# Create probability difference simulation class:
class ProbDiffSim(TwitterDataProcessor):

    # Inherit processed data from data prep class:
    def __init__(self, orient, frac_data=False, frac_start=None, frac_end=None):
        TwitterDataProcessor.__init__(self, orient, frac_data, frac_start, frac_end)
        self.rt_df = self.get_retweet_data()

        # Vectorized lambda function to count whether peer rating is greater than ego rating:
        self.count_more_extreme = np.vectorize(lambda x, y: 1 if y > x else 0)

        # Initializing dataframes that will be created during run of simulation:
        self.sim_df = None
        self.prob_diff_df = None

    # Run homophily simulation and append homophily peer ratings to new column in dataframe:
    def get_homophily_df(self):

        # Randomize order of dataset for simulation trial:
        homophily_df = get_random_users(self.rt_df)

        # Get all ego and peer ratings:
        ego_ratings = homophily_df['orig_rating_ego'].values
        peer_ratings = homophily_df['orig_rating_peer'].values

        # Initialize list of closest peers based on homophily simulation:
        closest_peers = []

        # For each ego rating in subset:
        for ego_rating in ego_ratings:
            # Find absolute differences:
            abs_diffs = np.abs(peer_ratings - ego_rating)

            # Find index of minimum absolute difference:
            min_diff_idx = abs_diffs.argmin()

            # Find closest peer at minimum difference index:
            closest_peer = peer_ratings[min_diff_idx]

            # Remove selected peer from peer pool:
            peer_ratings = np.delete(peer_ratings, min_diff_idx)

            # Append selected peer to closest peers list:
            closest_peers.append(closest_peer)

        # Create new column for closest peers based on homophily strategy:
        homophily_df['homoph_rating_peer'] = closest_peers

        return homophily_df

    # Creates dataframe with user id and probability differences after n trials:
    def get_sim_df(self):

        # Initialize dataframe:
        self.sim_df = pd.DataFrame()

        if self.orient == 'right':
            print('Beginning probability difference simulation for conservative users.', flush=True)

        elif self.orient == 'left':
            print('Beginning probability difference simulation for liberal users.', flush=True)

        if self.frac_data:
            print(f'Fractions chosen: {int(self.frac_start * 100)}% to {int(self.frac_end * 100)}%.', flush=True)

        # Run for 100 trials and continually add to the more_extreme_count dataframe:
        for i in range(100):
            print(f'Current iteration: {i + 1} of 100.', '\n', flush=True)

            start_time = time.time()

            # Get homophily ratings:
            homophily_df = self.get_homophily_df()

            # Define trial ego ratings:
            ego_ratings = homophily_df['orig_rating_ego']

            # Define trial peer ratings for each condition:
            peer_ratings_empi = homophily_df['orig_rating_peer']
            peer_ratings_homoph = homophily_df['homoph_rating_peer']

            # Create count column indicating whether peer is more extreme for each condition:
            homophily_df['is_more_extreme_homoph'] = self.count_more_extreme(ego_ratings, peer_ratings_homoph)
            homophily_df['is_more_extreme_empi'] = self.count_more_extreme(ego_ratings, peer_ratings_empi)

            # Append results to dataframe:
            self.sim_df = pd.concat([self.sim_df, homophily_df], axis=0, ignore_index=True)

            # Track trial runtime:
            minutes_taken = (time.time() - start_time) / 60
            print(f'Current iteration complete. Time elapsed: {minutes_taken: .2f} minutes.', '\n', flush=True)

        print('Simulation complete. Creating dataframe.', flush=True)

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
        self.prob_diff_df['poli_affil'] = np.repeat(self.orient, len(self.prob_diff_df))

        print('Dataframe created. Saving to csv.', flush=True)

    # Function to save dataframe:
    def save_prob_diff_df(self):
        data_path = 'data'

        if self.orient == 'left':
            path_beginning = os.path.join(data_path, 'prob_diff_left')

        elif self.orient == 'right':
            path_beginning = os.path.join(data_path, 'prob_diff_right')
        else:
            raise Exception('Political orientation must be defined as left or right.')

        if self.frac_data:
            file_path = path_beginning + f'_{self.frac_start}_{self.frac_end}.csv'
        else:
            file_path = f'{path_beginning}.csv'

        self.prob_diff_df.to_csv(file_path, index=False)
        print('Dataframe saved.', flush=True)

    def run(self):
        self.get_sim_df()
        self.get_prob_diff_df()
        self.save_prob_diff_df()


class MeanAbsDiffSim(TwitterDataProcessor):

    # Inherit processed data from data prep class:
    def __init__(self, orient, frac_data=False, frac_start=None, frac_end=None, thresholds=range(5, 45, 5)):
        TwitterDataProcessor.__init__(self, orient, frac_data, frac_start, frac_end)
        self.rt_df = self.get_retweet_data()
        self.thresholds = thresholds

        self.confints_random = list()
        self.confints_empi = list()

        # Initializing simulation dataframes:
        self.sim_df = None
        self.agg_sim_df = None

    def get_abs_diffs_df(self):

        abs_diff_df = get_random_users(self.rt_df)

        ego_ratings = abs_diff_df['orig_rating_ego'].values
        peer_ratings_empi = abs_diff_df['orig_rating_peer'].values
        peer_ratings_random = np.random.permutation(peer_ratings_empi)

        abs_diff_df['abs_diff_empi'] = np.abs(ego_ratings - peer_ratings_empi)
        abs_diff_df['abs_diff_random'] = np.abs(ego_ratings - peer_ratings_random)

        return abs_diff_df

    def get_sim_df(self):

        if self.orient == 'left':
            print('Beginning mean absolute difference simulation for liberal users.', flush=True)

        elif self.orient == 'right':
            print('Beginning mean absolute difference simulation for conservative users.', flush=True)

        else:
            raise Exception("Political orientation must be defined as left or right.")

        self.sim_df = pd.DataFrame()

        for threshold in self.thresholds:

            self.rt_df = self.rt_df[self.rt_df['rt'] >= threshold]

            print(f'Current threshold: {threshold} of {self.thresholds[-1]}', flush=True)

            threshold_df = pd.DataFrame()

            for i in range(100):

                abs_diff_df = self.get_abs_diffs_df()

                threshold_df = pd.concat([threshold_df, abs_diff_df], axis=0, ignore_index=True)

            threshold_df = threshold_df.groupby('userid', as_index=False)\
                .agg(mean_abs_diff_empi=('abs_diff_empi', 'mean'),
                     mean_abs_diff_random=('abs_diff_random', 'mean'))

            threshold_df['threshold'] = np.repeat(threshold, len(threshold_df))

            confint_empi = sms.DescrStatsW(threshold_df['mean_abs_diff_empi'].values).tconfint_mean()
            confint_random = sms.DescrStatsW(threshold_df['mean_abs_diff_random']).tconfint_mean()

            self.confints_empi.append(confint_empi)
            self.confints_random.append(confint_random)

            self.sim_df = pd.concat([self.sim_df, threshold_df], axis=0, ignore_index=True)

    def get_agg_sim_df(self):

        print('Simulation complete. Taking average results by threshold.', flush=True)
        self.agg_sim_df = self.sim_df.groupby('threshold', as_index=False).agg(
            mean_abs_diff_empi=('mean_abs_diff_empi', 'mean'),
            mean_abs_diff_random=('mean_abs_diff_random', 'mean'))

        self.agg_sim_df['confint_empi'] = self.confints_empi
        self.agg_sim_df['confint_random'] = self.confints_random

        self.agg_sim_df['poli_affil'] = np.repeat(self.orient, len(self.agg_sim_df))

    def save_agg_sim_df(self):

        print('Average results taken. Saving final dataframe.', flush=True)
        data_path = 'data'

        if self.orient == 'left':
            path_beginning = os.path.join(data_path, 'mean_abs_diff_left')

        elif self.orient == 'right':
            path_beginning = os.path.join(data_path, 'mean_abs_diff_right')
        else:
            raise Exception('Political orientation must be defined as left or right.')

        file_path = path_beginning + f'_{self.thresholds[0]}_{self.thresholds[-1]}.csv'

        self.agg_sim_df.to_csv(file_path, index=False)
        print('Dataframe saved.', flush=True)

    def run(self):
        self.get_sim_df()
        self.get_agg_sim_df()
        self.save_agg_sim_df()

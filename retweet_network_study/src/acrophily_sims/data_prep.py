"""
This script contains the TwitterDataProcessor class that produces
the processed dataframe used to complete all simulations.
"""

# Import ibraries:
import os
import numpy as np
import pandas as pd


# Define class TwitterDataProcessor:
class TwitterDataProcessor:
    """
    The TwitterDataProcessor class takes six parameters:

    1. poli_affil: the political affiliation ('left' or 'right') of users for which we wish to run the simulation
    2. frac_data: whether we wish to run the simulation on only a fraction of the data.
    3. frac_start: the starting percentage of users (in decimal form) if frac_data set to True
    4. frac_end: the ending percentage of users (in decimal form) if frac_data set to True
    5. users_file: The users ratings dataset CSV file path.
    6. rt_fle: The retweet network dataset CSV file path.

    There are three main functions:

    1. load_raw_data: This function loads the raw data for both the users ratings and retweet network datasets.
    2. preprocess_data: This function performs the following preprocessing steps for the data:
        - It subsets the users dataframe based on the chosen political affiliation on which to run the simulation.
        - It filters out both egos and peers with original tweet counts of less than 5 in both dataframes.
        - It filters out cases where egos retweeted themselves in the retweet dataframe.
        - It subsets down to the chosen fraction of users (if frac_data set to True) in the retweet dataframe.
    3. join_data: This function joins the users ratings dataset with the retweet dataset and suffixes to
    distinguish between ego and peer ratings.

    The main function get_retweet_data calls these functions in succession to create the final dataset.
    """

    def __init__(self, poli_affil, frac_data=False, frac_start=None, frac_end=None,
                 users_file=os.path.join('data', 'users_ratings.csv'), rt_file=os.path.join('data', 'rt_network.csv')):

        # Initialize political affiliation and data fraction attributes:
        self.poli_affil = poli_affil
        self.frac_data = frac_data
        self.frac_start = frac_start
        self.frac_end = frac_end

        # Initialize data files:
        self.users_file = users_file
        self.rt_file = rt_file

        # Initializing dataframes:
        self.users_df = pd.DataFrame()
        self.rt_df = pd.DataFrame()

    # Function to load raw data:
    def load_raw_data(self):

        print('Loading unprocessed user rating and retweet network datasets.', flush=True)

        # Load user ratings dataframe if file path exists:
        if os.path.exists(self.users_file):
            self.users_df = pd.read_csv(self.users_file)
            self.users_df = self.users_df.set_index('userid')
        else:
            return 'Users ratings file does not exist.'

        # Define retweet network dataframe if file path exists:
        if os.path.exists(self.rt_file):
            self.rt_df = pd.read_csv(self.rt_file)

        print('Datasets loaded. Processing and joining datasets.', flush=True)

    # Function to preprocess loaded raw data:
    def preprocess_data(self, min_tweets=5):

        # Subset to conservative ego ratings:
        if self.poli_affil == 'right':
            self.users_df = self.users_df[self.users_df['orig_rating'] > 0]

        # Subset to liberal ego ratings and convert ratings to positive scale:
        elif self.poli_affil == 'left':
            self.users_df = self.users_df[self.users_df['orig_rating'] < 0]
            self.users_df['orig_rating'] = self.users_df['orig_rating'] * -1

        # Subset based on min tweet threshold:
        self.users_df = self.users_df[self.users_df['orig_total_count'] >= min_tweets]

        # Subset retweet network ID to contain only egos and peers that meet min tweet threshold:
        userid_condition = self.rt_df['userid'].isin(self.users_df.index)
        rt_userid_condition = self.rt_df['rt_userid'].isin(self.users_df.index)
        self.rt_df = self.rt_df[userid_condition & rt_userid_condition]

        # Remove observations where user retweeted self
        self.rt_df = self.rt_df[self.rt_df['userid'] != self.rt_df['rt_userid']]

        # Subset fraction of users if needed to speed up simulation:
        if self.frac_data:

            # Get unique user ID values:
            all_users = np.unique(self.rt_df['userid'].values)

            # Subset to specified fraction of users:
            n_users_start = int(self.frac_start*len(all_users))
            n_users_end = int(len(all_users) * self.frac_end)
            users_fraction = all_users[n_users_start:n_users_end]

            # Return dataset with only user IDs in specified fraction:
            self.rt_df = self.rt_df[self.rt_df['userid'].isin(users_fraction)]

    # Function to join dataframes:
    def join_data(self):

        # Join on user ID and retweet user ID:
        self.rt_df = self.rt_df.join(self.users_df[['orig_rating']],
                                     on='userid').join(self.users_df[['orig_rating']],
                                                       on='rt_userid',
                                                       rsuffix='_peer')\
            .rename(columns={'orig_rating': 'orig_rating_ego'})

        print('Datasets joined. Data successfully loaded.', flush=True)

    # Main function that returns final dataframe:
    def get_retweet_data(self):
        self.load_raw_data()
        self.preprocess_data()
        self.join_data()

        return self.rt_df

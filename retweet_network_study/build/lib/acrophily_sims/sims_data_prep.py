import os
import numpy as np
import pandas as pd


class TwitterDataProcessor:

    def __init__(self, orient, frac_data=False, frac_start=None, frac_end=None):
        self.orient = orient
        self.frac_data = frac_data
        self.frac_start = frac_start
        self.frac_end = frac_end

        # Defining data file paths:
        self.data_path = 'data'
        self.users_file_path = os.path.join(self.data_path, 'users_ratings.csv')
        self.rt_network_file_path = os.path.join(self.data_path, 'rt_network.csv')

        # Initializing dataframes:
        self.users_df = None
        self.rt_df = None

    # Function to load raw data:
    def load_raw_data(self):

        print('Loading unprocessed user rating and retweet network datasets.', flush=True)
        # Load user ratings dataframe if file path exists:
        if os.path.exists(self.users_file_path):
            self.users_df = pd.read_csv(self.users_file_path)
            self.users_df = self.users_df.set_index('userid')
        else:
            return 'Users ratings file does not exist.'

        # Define retweet network dataframe if file path exists:
        if os.path.exists(self.rt_network_file_path):
            self.rt_df = pd.read_csv(self.rt_network_file_path)

        print('Datasets loaded. Processing and joining datasets.', flush=True)

    # Function to preprocess loaded raw data:
    def preprocess_data(self):
        # Set minimum number of tweets:
        min_tweets = 5

        # Subset to conservative ego ratings:
        if self.orient == 'right':
            self.users_df = self.users_df[self.users_df['orig_rating'] > 0]

        # Subset to liberal ego ratings and convert ratings to positive scale:
        elif self.orient == 'left':
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


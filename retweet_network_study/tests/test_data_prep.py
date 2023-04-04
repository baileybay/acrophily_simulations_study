"""
This script tests the data generating class TwitterDataProcessor.
It tests that data is returned at proper length and with proper
ratings based on subset into left or right-leaning users.
In both cases, ratings should be positive, with left-leaning
users' ratings being re-coded. It also assures proper fractioning
if frac_data is set to true.
"""

import os
import sys
import unittest
import numpy as np

# Find modules in proper directory:
appended_path = os.path.join('src', 'acrophily_sims')
sys.path.append(appended_path)

from acrophily_sims.data_prep import TwitterDataProcessor


class TestTwitterDataProcessor(unittest.TestCase):

    # Set up class:
    @classmethod
    def setUpClass(cls):
        print('setupClass')

    # Tear down class:
    @classmethod
    def tearDownClass(cls):
        print('teardownClass')

    # Set up instances to test:
    def setUp(self):
        # Test for left, right, and fraction cases:
        self.data_prep_left = TwitterDataProcessor(poli_affil='left',
                                                   users_file=os.path.join('data', 'test_users.csv'),
                                                   rt_file=os.path.join('data', 'test_rt.csv'))
        self.data_prep_right = TwitterDataProcessor(poli_affil='right',
                                                    users_file=os.path.join('data', 'test_users.csv'),
                                                    rt_file=os.path.join('data', 'test_rt.csv'))
        self.data_prep_frac = TwitterDataProcessor(poli_affil='left', frac_data=True,
                                                   frac_start=0.0, frac_end=0.99,
                                                   users_file=os.path.join('data', 'test_users.csv'),
                                                   rt_file=os.path.join('data', 'test_rt.csv'))

    # Tear down instances:
    def tearDown(self):
        print('tearDown')

    # Get test cases for unit tests:
    def get_test_cases(self):
        # Gets test cases for preprocessing step:
        self.data_prep_left.load_raw_data()
        self.data_prep_left.preprocess_data()

        self.data_prep_right.load_raw_data()
        self.data_prep_right.preprocess_data()

        self.data_prep_frac.load_raw_data()
        self.data_prep_frac.preprocess_data()

    # Test that ego ratings are filtered/coded as positive:
    def assert_positive_ratings(self):
        # Define left and right user dataframes:
        left_users_df = self.data_prep_left.users_df
        right_users_df = self.data_prep_left.users_df

        # Assert user ratings filtered (and converted for left) to positive scale:
        self.assertTrue(all(left_users_df['orig_rating'] > 0))
        self.assertTrue(all(right_users_df['orig_rating'] > 0))

    # Assert that proper fraction of users appear in fraction dataset:
    def assert_proper_frac_df_len(self):
        # Define retweet dataframe for fraction of left-wing users:
        frac_left_rt_df = self.data_prep_frac.rt_df

        # Get number of unique users in left-wing fraction dataframe:
        n_users_frac_left = len(np.unique(frac_left_rt_df['userid']))

        # Define fraction end of left-wing fraction dataframe:
        frac_left_end = self.data_prep_frac.frac_end

        # Get unique number of users in full left-wing retweet dataframe:
        left_rt_df = self.data_prep_left.rt_df
        n_users_left = len(np.unique(left_rt_df['userid']))

        # Assert that proper fraction of users represented in fraction dataframe:
        self.assertEqual(n_users_frac_left, int(frac_left_end * n_users_left))

    # Test loading raw data (all instances work same):
    def test_load_raw_data(self):
        # Tests that file exists:
        self.assertTrue(os.path.exists(self.data_prep_left.users_file))
        self.assertTrue(os.path.exists(self.data_prep_left.rt_file))

    # Test output of processing and joining steps for three instances:
    def test_get_retweet_data(self):
        # Get test cases:
        self.get_test_cases()

        # Test positive ratings for both left and right after processing:
        self.assert_positive_ratings()

        # Test number of users in fraction dataset is correct:
        self.assert_proper_frac_df_len()

        # Join left-wing dataframes:
        self.data_prep_left.join_data()

        # Assert that a non-zero length dataframe exists:
        self.assertTrue(len(self.data_prep_left.rt_df) > 0)

        # Assert non-zero length dataframe exists for fraction dataframe:
        self.data_prep_frac.join_data()
        self.assertTrue(len(self.data_prep_frac.rt_df) > 0)


if __name__ == '__main__':
    unittest.main()

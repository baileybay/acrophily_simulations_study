"""
This script tests the main functions that run the acrophily simulation.
It assures proper dataframe length and number of columns.
"""

import os
import sys
import unittest
import numpy as np

# Find modules in proper directory:
appended_path = os.path.join('src', 'acrophily_sims')
sys.path.append(appended_path)

from acrophily_sims.sims import AcrophilySim


class TestProbDiffSim(unittest.TestCase):

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
        self.sim_left = AcrophilySim(poli_affil='left',
                                       users_file=os.path.join('data', 'test_users.csv'),
                                       rt_file=os.path.join('data', 'test_rt.csv'),
                                     thresholds=range(1, 2))

        self.sim_right = AcrophilySim(poli_affil='right',
                                        users_file=os.path.join('data', 'test_users.csv'),
                                        rt_file=os.path.join('data', 'test_rt.csv'),
                                      thresholds=range(1, 2))

        self.sim_frac = AcrophilySim(poli_affil='left',
                                     users_file=os.path.join('data', 'test_users.csv'),
                                     rt_file=os.path.join('data', 'test_rt.csv'),
                                     frac_data=True, frac_start=0.0, frac_end=0.99,
                                     thresholds=range(1, 2))

    # Tear down instances:
    def tearDown(self):
        print('tearDown')

    def get_test_case_data(self):
        self.sim_left.rt_df = self.sim_left.get_retweet_data()
        self.sim_right.rt_df = self.sim_right.get_retweet_data()
        self.sim_frac.rt_df = self.sim_frac.get_retweet_data()

    # Function to call test cases for homophily df:
    def get_test_cases_acrophily_df(self):
        self.get_test_case_data()

        self.sim_left.get_acrophily_df()
        self.sim_right.get_acrophily_df()
        self.sim_frac.get_acrophily_df()

    def get_test_cases_sim_df(self):
        self.get_test_case_data()

        self.sim_left.get_sim_df()
        self.sim_right.get_sim_df()
        self.sim_frac.get_sim_df()

    def assert_proper_len(self, df_name):

        # Assert aggregation of threshold df by unique user ID:
        if df_name == 'agg_threshold_df':
            df_len_left = len(self.sim_left.agg_threshold_df)
            df_len_right = len(self.sim_right.agg_threshold_df)

            n_users_left = len(np.unique(self.sim_left.threshold_sim_df['userid'].values))
            n_users_right = len(np.unique(self.sim_right.threshold_sim_df['userid'].values))

            self.assertEqual(df_len_left, n_users_left)
            self.assertEqual(df_len_right, n_users_right)

        # Assert length of final df is same as number of thresholds:
        elif df_name == 'agg_sim_df':
            df_len_left = len(self.sim_left.agg_sim_df)
            df_len_right = len(self.sim_right.agg_sim_df)

            n_thresholds_left = len(np.unique(self.sim_left.sim_df['threshold']))
            n_thresholds_right = len(np.unique(self.sim_right.sim_df['threshold']))

            self.assertEqual(df_len_left, n_thresholds_left)
            self.assertEqual(df_len_right, n_thresholds_right)

    def test_get_sim_df(self):
        self.get_test_cases_sim_df()
        self.assert_proper_len(df_name='agg_threshold_df')


if __name__ == '__main__':
    unittest.main()
"""
This script contains tests for the prob diff sim specifically.
It assures that the data returned at each step of the simulation
contains the proper number of columns and that the dataframe lengths
are as expected based on the state of the simulation.
"""

import os
import sys
import unittest
import numpy as np

# Find modules in proper directory:
appended_path = os.path.join('src', 'acrophily_sims')
sys.path.append(appended_path)

from acrophily_sims.sims import ProbDiffSim


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
        self.sim_left = ProbDiffSim(poli_affil='left',
                                    users_file=os.path.join('data', 'test_users.csv'),
                                    rt_file=os.path.join('data', 'test_rt.csv'))
        self.sim_right = ProbDiffSim(poli_affil='right',
                                     users_file=os.path.join('data', 'test_users.csv'),
                                     rt_file=os.path.join('data', 'test_rt.csv'))
        self.sim_frac = ProbDiffSim(poli_affil='left', frac_start=0.0, frac_end=0.99,
                                    users_file=os.path.join('data', 'test_users.csv'),
                                    rt_file=os.path.join('data', 'test_rt.csv'))

    # Tear down instances:
    def tearDown(self):
        print('tearDown')

    def get_test_case_data(self):
        self.sim_left.rt_df = self.sim_left.get_retweet_data()
        self.sim_right.rt_df = self.sim_right.get_retweet_data()
        self.sim_frac.rt_df = self.sim_frac.get_retweet_data()

    # Function to call test cases for homophily df:
    def get_test_cases_homophily_df(self):
        self.get_test_case_data()

        self.sim_left.get_homophily_df()
        self.sim_right.get_homophily_df()
        self.sim_frac.get_homophily_df()

    # Function to call test cases for sim df:
    def get_test_cases_sim_df(self):
        self.get_test_case_data()

        self.sim_left.run_sim()
        self.sim_right.run_sim()
        self.sim_frac.run_sim()

    # Function to call test cases for prob diff df:
    def get_test_cases_prob_diff_df(self):
        self.get_test_case_data()

        # Get sim df for three test cases:
        self.get_test_cases_sim_df()

        self.sim_left.get_prob_diff_df()
        self.sim_right.get_prob_diff_df()
        self.sim_frac.get_prob_diff_df()

    # Function that asserts proper proportionality between dataframes:
    def assert_df_proportions(self, df_name):

        # Test that homophily df is 70% subset
        if df_name == 'homophily_df':

            # Tests that 70% subset works within 1 whole number:
            self.assertAlmostEqual(len(self.sim_left.homophily_df), 0.7 * len(self.sim_left.rt_df), delta=1)
            self.assertAlmostEqual(len(self.sim_right.homophily_df), 0.7 * len(self.sim_right.rt_df), delta=1)
            self.assertAlmostEqual(len(self.sim_frac.homophily_df), 0.7 * len(self.sim_frac.rt_df), delta=1)

        # Test that sim_df is product of 1000 iterations over homophily df (give or take 100 values):
        elif df_name == 'sim_df':
            self.assertAlmostEqual(len(self.sim_left.sim_df), 1000 * len(self.sim_left.homophily_df), delta=100)
            self.assertAlmostEqual(len(self.sim_right.sim_df), 1000 * len(self.sim_right.homophily_df), delta=100)
            self.assertAlmostEqual(len(self.sim_frac.sim_df), 1000 * len(self.sim_frac.homophily_df), delta=100)

    def assert_no_duplicates(self):
        # Test that homophily dataframe didn't duplicate every iteration:
        no_dups_sim_left = self.sim_left.sim_df.drop_duplicates()
        no_dups_sim_right = self.sim_right.sim_df.drop_duplicates()
        no_dups_sim_frac = self.sim_frac.sim_df.drop_duplicates()

        self.assertFalse(len(no_dups_sim_left) == len(self.sim_left.homophily_df))
        self.assertFalse(len(no_dups_sim_right) == len(self.sim_right.homophily_df))
        self.assertFalse(len(no_dups_sim_frac) == len(self.sim_frac.homophily_df))

    def test_get_homophily_df(self):

        # Get test cases for function:
        self.get_test_cases_homophily_df()

        # Assert that homophily dataframe is 70% of original dataframe (within rounding):
        self.assert_df_proportions(df_name='homophily_df')

    # Test simulation dataframe:
    def test_get_sim_df(self):

        # Get sim df for three test cases:
        self.get_test_cases_sim_df()

        # Test that sim df roughly 100 times larger than homophily df:
        self.assert_df_proportions(df_name='sim_df')

        # Test that homophily dataframe didn't duplicate every iteration:
        self.assert_no_duplicates()

    def test_get_prob_diff_df(self):

        # Get prob diff df for three test cases:
        self.get_test_cases_prob_diff_df()

        # Assert prob diff df has single row for each user in sim df:
        n_users_sim_left = len(np.unique(self.sim_left.sim_df['userid']))
        n_users_sim_right = len(np.unique(self.sim_right.sim_df['userid']))
        n_users_sim_frac = len(np.unique(self.sim_frac.sim_df['userid']))

        self.assertEqual(len(self.sim_left.prob_diff_df), n_users_sim_left)
        self.assertEqual(len(self.sim_right.prob_diff_df), n_users_sim_right)
        self.assertEqual(len(self.sim_frac.prob_diff_df), n_users_sim_frac)


if __name__ == '__main__':
    unittest.main()

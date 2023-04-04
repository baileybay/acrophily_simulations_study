"""
- Script to run Bayesian hierarchical logistic regression for acrophily simulation
- Predicts whether peer is more extreme based on sim condition, controlling for retweet number (aka threshold)
- Intercepts vary by user and by retweet threshold
"""
# Import libraries:
from argparse import ArgumentParser
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymc3 as pm
from sklearn.preprocessing import StandardScaler


# Parse arguments and pass to arguments object:
def get_args(args=None):

    # Initialize argument parser for command line:
    parser = ArgumentParser(prog='Simulation Data Processor')
    
    # Create command for data path:
    parser.add_argument('-d', '--data_path', default="data",
                        help="Pathway to the data files to process (default: data")


    # Create command for political affiliation:
    parser.add_argument('-p', '--poli_affil', default="left",
                        help="Political affiliation subset you wish to run simulation on (left/right)")

    return parser.parse_args(args=args)


# Script to load and process data for model:
def load_data(data_path, poli_affil=None):
    
    # Label encode based on sim condition:
    def label_encode(sim_condition):
        
        if sim_condition == 'orig_rating_peer':
            return '0'
        elif sim_condition == 'homoph_rating_peer':
            return '1'
        elif sim_condition == 'acroph_rating_peer_min':
            return '2'
        else:
            return '3'
    
    # Read in data:
    file_name = f'acrophily_sim_{poli_affil}_indiv.csv'
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    
    # Filter threshold:
    df = df[df['threshold'] <= 5]
    
    # Convert user ID to integer:
    df['userid'] = df['userid'].astype(int)
    
    # Convert to long format:
    df = pd.melt(df, id_vars=['userid', 'threshold', 'orig_rating_ego'], 
                 value_vars=['orig_rating_peer', 'homoph_rating_peer', 
                             'acroph_rating_peer_min', 'acroph_rating_peer_max'])
    
    # Rename columns:
    df = df.rename(columns={'variable': 'sim_condition', 'value': 'peer_rating'})
    
    # Create dummy variable indicating whether peer is more extreme:
    df['more_extreme'] = np.where(df['peer_rating'] > df['orig_rating_ego'],
                                 1, 0)
    
    # Set random seed and subset data down to random fraction of users:
    np.random.seed(109)
    users = np.unique(df['userid'])
    n_users = len(users)
    random_users = np.random.choice(users, size=int(0.5*n_users), replace=False)
    df = df[df['userid'].isin(random_users)]
 
    # Factorize user ID to create index for each user:
    df['userid_idx'] = pd.factorize(df['userid'])[0]
    
    # Label encode sim condition variable:
    df['sim_condition'] = df['sim_condition'].apply(label_encode)
    
    # Convert sim condition variable to dummy columns:
    df = pd.get_dummies(df, drop_first=True)
    
    # Rename columns:
    df = df.rename(columns={'sim_condition_1': 'homophily', 
                            'sim_condition_2': 'acrophily_min',
                           'sim_condition_3': 'acrophily_max'})
    
    # Scale threshold value:
    sc = StandardScaler()
    df['threshold'] = sc.fit_transform(df['threshold'].values.reshape(-1, 1))
    
    return df


# Run Bayesian hierarchical model for logistic regression:
def run_model(df):
    
    # Get coords:
    coords = {"user": np.unique(df['userid_idx']),
     "obs_id": np.arange(len(df))}

    # Run model:
    with pm.Model(coords=coords) as model:
    
        # Predictor variables:
        homoph = df['homophily']
        acroph_min = df['acrophily_min']
        acroph_max = df['acrophily_max']
        thresh = df['threshold']

        # Indexes for varying intercepts for each user within each threshold:
        user_idx = pm.Data("user_idx", df['userid_idx'], dims="obs_id")

        # Mean and tau for varying intercepts:
        mu_alpha = pm.Normal('mu', mu=0, sd=100)
        tau_alpha = pm.Gamma('tau', alpha=0.1, beta=0.1)

        # Initialize varying intercepts with dims for each user:
        alpha = pm.Normal('alpha', mu=mu_alpha, tau=tau_alpha, dims="user")
        
        # Generate varying intercepts for each user:
        intercepts = alpha[user_idx]

        # Generate coefficients with uniform prior within -100 and 100:
        b_homoph = pm.Uniform('homoph', lower=-100, upper=100)
        b_acroph_min = pm.Uniform('acroph_min', lower=-100, upper=100)
        b_acroph_max = pm.Uniform('acroph_max', lower=-100, upper=100)
        b_thresh = pm.Uniform('thresh', lower=-100, upper=100)

        # Run logistic regression model with sigmoid activation function:
        p = pm.Deterministic('p', pm.math.sigmoid(intercepts + b_homoph*homoph + \
                                                  b_acroph_min*acroph_min + \
                                                  b_acroph_max*acroph_max + \
                                                  b_thresh*thresh))

        # Generate observations using model and actual outcomes:
        obs = pm.Bernoulli('obs', p=p, observed=df['more_extreme'])

        # Get trace to analyze results:
        trace = pm.sample(1000, tune=1000, target_accept=0.95)
        
    return trace


# Save trace plot:
def save_trace_plot(trace, file_name, fig_path='figures'):
    
    az.plot_trace(trace, var_names=['homoph', 'acroph_min', 'acroph_max', 'thresh'],
                 compact=True, figsize=(16, 14))
    plt.savefig(os.path.join(fig_path, file_name))
    

# Save model summary DF:
def save_summary_df(trace, data_path, file_name):
    
    file_path = os.path.join(data_path, file_name)
    
    df = az.summary(trace, var_names=['homoph', 'acroph_min', 'acroph_max', 'thresh'])
    df.to_csv(os.path.join(data_path, file_name))
    

# Main function to load data, run model and save results:
def main():
    args = get_args()
    df = load_data(data_path=args.data_path, poli_affil=args.poli_affil)
    trace = run_model(df)
    save_trace_plot(trace=trace, file_name=f'trace_plot_{args.poli_affil}.jpg')
    save_summary_df(trace=trace, data_path=args.data_path, file_name=f'bayes_model_{args.poli_affil}.csv')
    

if __name__ == '__main__':
    main()

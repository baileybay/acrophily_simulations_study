# Acrophily Simulations Study Code

Here, you will find a directory called retweet_network_study that contains code I produced in collaboration with Harvard Business School's Dr. Amit Goldenberg and USC's Dr. Goran Muric. My Python code was adapted from the analyze_and_plot.ipynb Jupyter Notebook found in the source_code directory first written by Goran Muric. All R code was written in collaboration with Amit Goldenberg. The rest of the code was written and/or adapted by me to produce a working Python package to run three simulations used to conduct a study on online network formation, as well as code to process data generated from the simulations to then analyze using a Bayesian model and to visualize using ggplot. Further, tests were written for the simulation package code, and a GitHub Actions workflow was created to set up a pipenv environment and run all tests when pushing to the main branch. Finally, a Dockerfile was created to run the simulations from a Docker container. 

Results indicate that both liberals and conservatives on Twitter prefer to retweet peers who are both copartisan and more extreme than themselves, relative to what we would expect if they were retweeting by chance or retweeting based only on political similarity. This highlights a key potential driver of polarization in the modern age, perhaps paving the way for a solution that helps reduce political conflict in America. 

The preprint can be found at the following link:
https://osf.io/cmx4p/

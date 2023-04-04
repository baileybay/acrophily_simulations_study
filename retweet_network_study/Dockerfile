FROM python:3.9-slim

ENV VAR1=10

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install wget and remove unnecessary folder:
RUN  apt-get update \
  && rm -rf /var/lib/apt/lists/* 

# Install pipenv:
RUN pip install pipenv
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy directory into container and set it as working directory:
COPY . /retweet_network_study
WORKDIR /retweet_network_study

# Install dependencies with pipenv:
RUN pipenv install --deploy --system

# Creates a non-root user and adds permission to access acrophily sims folder:
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /retweet_network_study
USER appuser

# Run acrophily simulation module command:
CMD ["python", "-m", "acrophily_sims"]

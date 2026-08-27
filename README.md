# Customer Churn Prediction

Live app: https://customer-churn-prediction-rahul.streamlit.app/

## What this project is

This is an end to end applied machine learning project where I took a customer churn prediction model all the way from raw data to a live, working application that anyone can open and try. I built this specifically to go beyond just training a model in a notebook, since a model that only exists in a Jupyter notebook can't actually be used by anyone else. I wanted something I could point to and say here, try it yourself, and have it just work.

## The business question

Given how often a customer has purchased and how much they've spent, can I predict whether they're likely to stop buying from this business, and can I turn that prediction into something a real business could actually use to prioritize who to reach out to.

## Where the data comes from

I reused the cleaned Online Retail II dataset from an earlier project of mine, a customer segmentation and retention analysis I built using RFM analysis and K-means clustering. That project is here if you want to see how the data was originally cleaned: https://github.com/Rahulkulkarni2002/customer-segmentation-retention-analysis

Since I already had a clean, well understood dataset with Recency, Frequency, and Monetary features already built for each customer, it made sense to reuse it here rather than start over with a brand new dataset. The full dataset covers about 804,000 transactions across roughly 5,855 customers.

## How I defined churn

I didn't have a churn label handed to me, so I had to define one myself. I used Recency, which is how many days it's been since a customer's last purchase, and defined churn as any customer who hasn't purchased in the last 90 days. I picked 90 days deliberately rather than an arbitrary number. The median Recency across all customers was already around 95 days, so a 90 day cutoff gave me a genuinely balanced target, about 50.7 percent churned and 49.3 percent active, rather than something heavily skewed where almost everyone or almost no one is labeled churned. A shorter cutoff like 30 days would have classified most of the customer base as churned, which wouldn't have been a useful label for a business to act on.

## Features and a leakage mistake I avoided

For features I engineered several things from the raw transaction data. Frequency and Monetary came directly from the existing RFM table. I also added Tenure, which is the number of days between a customer's first and last purchase, calculated straight from the transaction dates. I added AvgOrderValue, which is Monetary divided by Frequency, since spend per order is a genuinely different signal than either raw number alone. I log transformed Frequency and Monetary since both were right skewed, and I added an interaction term multiplying Frequency and Monetary together, so the models could see combined high frequency, high spend behavior directly.

One thing I want to call out specifically is that I deliberately did not include Recency as a feature in the model. Recency is literally what I used to create the Churned label in the first place, so if I'd included it as a feature, the model would essentially just be learning the rule I used to build the label, which isn't real prediction, it's circular. I checked for this kind of leakage before training anything, not after.

## Model comparison

I didn't just pick one model and go with it. I trained and compared six different approaches on the same train test split, using the same features, so the comparison would actually be fair.

First I tried a simple two feature version using just Frequency and Monetary, comparing Logistic Regression against XGBoost. XGBoost barely beat the simple linear model, which told me there wasn't much complex nonlinear signal for a more sophisticated model to exploit with just two features.

Then I expanded to the full seven feature set described above and tried Logistic Regression, XGBoost, a neural network using scikit learn's MLPClassifier, and Random Forest. This time the richer features actually helped every model improve, and the differences between models became more meaningful.

I also tried combining models into a voting ensemble, first with three models and then with four, including Random Forest. The four model ensemble ended up performing almost identically to the three model version, which told me Random Forest wasn't adding much new information on top of what the other models were already capturing.

Here's the honest comparison across everything I tried, measured on a held out test set.

Logistic Regression with two features got 69.0 percent accuracy and an F1 score of 0.73.

XGBoost with two features got 69.4 percent accuracy and an F1 score of 0.72.

Logistic Regression with seven features got 71.0 percent accuracy and an F1 score of 0.72.

XGBoost with seven features got 71.9 percent accuracy and an F1 score of 0.74.

The neural network with seven features got 72.2 percent accuracy and an F1 score of 0.75.

Random Forest with seven features got 72.4 percent accuracy and an F1 score of 0.749, which turned out to be the best F1 score and best recall of any single model I tried.

The three model ensemble got 72.8 percent accuracy with an F1 score of 0.746.

The four model ensemble, adding Random Forest into the mix, got 72.8 percent accuracy with an F1 score of 0.747, essentially no improvement over the simpler three model version.

In the end I chose Random Forest as the model I actually deploy, not because it's the most complex option, but because it matched or beat every more complicated alternative, including the ensembles, while being much simpler to actually put into production. That felt like the right tradeoff for a real deployed system rather than chasing a marginal, possibly noisy improvement from added complexity.

## Experiment tracking

I used MLflow to track all six models I trained, logging the parameters, accuracy, precision, recall, and F1 score for each one, along with saving the actual trained model files. I ran into a couple of real environment issues along the way, MLflow initially refused to save the XGBoost and neural network models because of a security check on untrusted object types, which I resolved by using XGBoost's own dedicated logging function and explicitly marking the specific neural network internal type as trusted. I also had the MLflow web dashboard fail to show my logged runs for a while, which turned out to be a mismatch between where the tracking data was actually saved and where the dashboard was looking, not a problem with my actual training code. I confirmed this directly using MLflow's Python client rather than just trusting the browser, and found all my runs were genuinely saved correctly the whole time.

## Building the API

I built a FastAPI application that loads and trains the Random Forest model once when the server starts, then exposes a predict endpoint that takes in a customer's Frequency, Monetary, and Tenure, calculates the same engineered features used in training, and returns both a churn prediction and the underlying probability. I added input validation so that invalid input, like a Frequency of zero which would cause a divide by zero error when calculating average order value, returns a clean error message instead of crashing the server. I tested this using FastAPI's automatic interactive documentation page and confirmed both the validation and the actual predictions work correctly.

## The deployed app

For the live, publicly deployed version, I built a self contained Streamlit application rather than running the API and the interface as two separate connected services. The model is trained once and cached using Streamlit's caching decorator, so it doesn't retrain on every single interaction. Someone visiting the live app can type in a customer's order count, total spend, and how long they've been a customer, click a button, and immediately see a churn prediction along with the probability behind it.

I made this choice deliberately after thinking about whether I actually needed two separate running services for this to be a real, legitimate applied ML project. The FastAPI version still exists in this repository and demonstrates that I can build a real API that other software could call, but for the version anyone can actually visit and use right now, a single self contained app was simpler to deploy and just as complete from a user's perspective.

## What I'd do differently with more time

I'd like to go back and test whether adding more features, maybe something about product category preferences or geographic location, actually gives the more complex models like the neural network or the ensembles a real advantage over Random Forest, rather than the marginal differences I found here. I'd also like to properly connect the deployed app to the MLflow tracked model artifacts instead of retraining directly at startup, since that's closer to how a real production system would actually work.

## Files in this repository

churn.ipynb contains the full analysis, from loading the data through training and comparing all six models.

main.py is the FastAPI application, meant to be run locally with uvicorn.

app.py is the original two service version of the Streamlit app that calls the FastAPI endpoint over the network, kept here to show that architecture works.

streamlit_app.py is the self contained version actually deployed live, with the model built directly into the app.

requirements.txt lists the Python packages needed to run this project.

cleaned_retail.csv and rfm_table.csv are the underlying cleaned dataset and RFM feature table, reused from the companion segmentation project.

## Tools and skills used

Python, pandas, numpy, scikit learn, XGBoost, MLflow, FastAPI, Streamlit, and Streamlit Community Cloud for deployment.

# Natural Language Processing Using Subreddits


## Contents

- [Problem_Statement](#Problem-Statement)
- [Summary of project](#Summary-of-project)
- [EDA](#EDA)
- [Modeling](#Modeling)
- [Visualization](#Visualization)
- [Conclusions](#Conclusions)


## Problem Statement

Collect data from two different subreddits from the popular reddit website and build a model to predict the subreddit


#### Summary of project:
We had to collect data from reddit. The first challenge was to select two subreddits that were somewhat similar but different enough to create a good model for it. We chose Math and Physics. We got about 80 thousand posts from Math but Physics only had 40 thousand. Luckily we also found the AskPhysics subreddit that had 40 thousand more posts so we combined those two. After we collected the data, we combined all posts and created models to test how good our models did in predicting which post came from which subreddit.

#### EDA:
We cleaned the data, joined the dataframes, got rid of nulls, changed the target column to 1s and 0s, made sure we had no duplicates and analized if the data was good and ready to try modeling.

We considered lemmatizing but with math and physics, some technical words might be affected and we had already decided to use Count Vectorizer and Tfid Vectorizer which already tokenize and get rid of punctuation and do other helpful cleaning. We also considered the emoji dilemma but concluded the emojis also give information and that they probably get used in the model by their text unicode identificator so we did not worry about them.

#### Modeling:
We tried Count Vectorizer and Tfid Vectorizer with several different models using pipelines to try several different parameters. We tried Multinomial Naive Bayes, Random Forest, Extra Trees and also Voting Classifier with the previous models. We got different scores for different models but overall we had over 90% accuracy on our models which was pretty good.

#### Visualizations:
We used some visualizations to see the different model's performances and also the different parameters for our best model.

#### Conclusions:
We gathered a little more data from reddit just to predict using our best model and found out that our model did very good on this exclusive never before seen data. Overall a great result.
Other things to consider if we had more time:
Sentiment analysis
Play with less data
Check if we can predict the year a post came from
Science website recommendation for advertisement 

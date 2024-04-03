# Entity Resolution Project README

## Table of Contents
1. [Overview](#overview)
2. [Tools Used](#tools-used)
3. [Data Cleaning](#data-cleaning)
4. [Classifier Building](#classifier-building)
5. [Database Setup](#database-setup)
6. [Training Data Simulation](#training-data-simulation)
7. [Deployment](#deployment)


## Overview
This project focuses on entity resolution, specifically mapping grants to doctors from multiple datasets. The goal is to build a classifier that can predict matches between grants and doctors using various features such as the Jaro-Winkler distance between last names and using word embeddings from huggingface models and fasttext. The process involves reading in data, cleaning and preprocessing it, building and training a classifier (which includes simulating initial training data), setting up a database and creating tables to store connections, and deploying the classifier on testing data for matching purposes.

## Tools Used
- Python (programming language)
- Pandas (data manipulation)
- Scikit-learn (machine learning)
- XGBoost (Classifier)
- SQLite (database)
- Git (version control)
- GitHub (code hosting and collaboration)

## Data Cleaning
The data cleaning phase involves preprocessing both the grants and doctors datasets. Tasks include handling missing values, standardizing formats (e.g., names, dates), and extracting relevant from the datasets for matching purposes. Specifically, various dates were imputed and sub selection of columns were chosen for the classfier.

## Classifier Building
We built a classifier using machine learning techniques to predict matches between grants and doctors. Features such as Jaro-Winkler distance between last names, matching city names, and the degrees of spearation between embeddings were used. We instanitated an XGBoost classifier, simulated training data by sampling from common names and hand labelling matches, then trained and evaluated our model.

## Database Setup
We set up an SQLite database to store our data and establish connections between grants and doctors. This database allows for efficient querying and retrieval of matched entities. We also set up bridge tables to house potential matches (doctors and grants with the same "last name" feature, for example).

## Training Data Simulation
To train our classifier, we simulated training data by generating positive and negative samples of matched and unmatched pairs of grants and doctors. This simulated data helps improve the classifier's accuracy and generalization. The data simulation process can be found in the data_simulator file within the program_files.distance_classifier directory.

## Deployment
The trained classifier is deployed to perform real-time matching between grants and doctors. The deployment can find matches between grants and doctors to analyze how and by who doctors are recieving money.

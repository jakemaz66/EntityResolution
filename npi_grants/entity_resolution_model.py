import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import numpy
import os
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score
from data_readers import grants


class EnttityResolver():

    def __init__(self, model_dir):
        """
        Constructor

        Args:
        model is the type of classifier we are loading
        model_dir is the directory in which to save the model
        """
        self.model = self._initialize_xgb_model()
        self.metadata = {}
        self.model_dir = model_dir
        

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """
        Train the classifier on the data

        Args:
        features: The feature columns of the dataset
        labels: The column that is to be predicted
    
        """
        #Splitting the data
        features, features_test, labels, labels_test = train_test_split(features, labels, test_size=0.2)

        self.model.fit(features, labels.astype(float))

        #Updating the metadata
        self.metadata['training_data'] = datetime.datetime.now().strftime('%Y%m%d')
        self.metadata['training_rows'] = len(features)
        self.metadata['accuracy'] = self.assess(features_test, labels_test)


    def predict(self, features: pd.DataFrame, proba: bool = False):
        """
        Model predicts on the test_data

        Args:
        features: the features of the dataset
        proba: whether to return probabilities

        Returns:
        numpy array 
        """
        
        #Adding in check for training
        if len(self.metadata) == 0:
            return ValueError("Model must be trained first")

        if proba == True:
            self.model.predict_proba(features)[:, 0]

        return self.model.predict(features)
        
    
    def save(self, file_name: str, overwrite: bool = False):
        """
        Save to location path on hard drive

        Args:
        file_name: the name of the file to save
        overwrite: a boolean to check for permission to overwrite an existing file_name
        """
        if (len(self.metadata) == 0):
            return ValueError("Model must be trained before saving")
        
        #Adding in date to the filename
        now = datetime.datetime.now().strftime('%Y%m%d')
        if file_name[:6] != now:
            file_name = f'{now}_{file_name}'

        #Checking if extension of file is correct format
        if os.path.splitext(file_name)[1] != '.json':
            file_name = file_name + '.json'

        path = os.path.join(self.model_dir, file_name)
        metadatapath = os.path.splitext(path)[0] + '_metadata.json'

        if not overwrite and (os.path.exists(path) or os.path.exists(metadatapath)):
            return ValueError("Cannot overwrite existing file")
        
        self.model.save_model(path)

        with open(metadatapath, 'w') as file:
            json.dump(self.metadata, file)


        
    def load(self, file_name):
        """Load in a model filename with associated metadata from 
        model_dir
        """
        path = os.path.join(self.model_dir, file_name)
        metadata_path = os.path.splitext(path)[0] + '_metadata.json'
        self.model.load_model(path)

        with open(metadata_path) as file:
            self.metadata = json.load(file)
        
        
    def assess(self, features, labels) -> float:
        """
        Returns the accuracy of the model

        Args:
        features: The feature columns of the dataset
        labels: The column that is to be predicted

        Returns:
        Accuracy of model chosen metric
        
        """
        preds = self.predict(features, proba=False)
        binary_labels = []

        #pred_labels returns probabilities in numpy array, using thresholds to make integer predictions
        for i in preds:
            if i > 0.5:
                binary_labels.append(1)
            else:
                binary_labels.append(0)

        return accuracy_score(labels, binary_labels)

    def _initialize_xgb_model(self):
        """Create a new xgbclassifier"""
        return xgb.XGBClassifier()
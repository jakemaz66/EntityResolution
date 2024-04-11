import numpy as np
import pandas as pd
import hnswlib 
import fasttext
from npi_grants import create_features, entity_resolution_model
from sklearn.neighbors import NearestNeighbors
from npi_grants.data_readers import npi, grants

from npi_grants import hnsw
from npi_grants.sql import db


class Predict:
    """A class for predicting matches between grantees and doctors"""

    def __init__(self, providers: pd.DataFrame, classifier, classifier_load_path: str,
                 hnsw_instance, batch_size: int, embed_model=fasttext.load_model('npi_grants/data/cc.en.50.bin')):
        
        #Declaring attributes
        self.providers = providers
        self.hnsw_instance = hnsw_instance
        self.embed_model = embed_model
        self.batch_size = batch_size
        self.features_creator = create_features.CreateFeatures()

        classifier.load(classifier_load_path)
        self.model = classifier


    def get_index_labels(self, k_neighbors: int, grantees):
        """This function gets the neighbors for each grantee and returns two dataframes for comparison"""

        #Calculating a fullname columns
        grantees['fullname'] = (grantees['forename'].apply(lambda x: x.lower()) + " " + 
        grantees['last_name'].apply(lambda x: x.lower()))

         #Embedding the fullname using fasttext
        grantees['vector_g'] = grantees['fullname'].apply(
            lambda x: self.embed_model.get_sentence_vector(x) if not pd.isnull(x) else None)

        #Getting the HNSW (provider) indices
        index =  self.hnsw_instance.get_indices()

        #Chunking grantee dataframe
        grantees = grantees.iloc[:self.batch_size, :]
        grantee_array = np.array(grantees['vector_g'].dropna().tolist())

        label = []

        #Performing KNN queries
        for i in grantees.index:
            labels, _ = index.knn_query(grantee_array[i], k=k_neighbors)
            label.append(labels)
        
        #Getting list of indices and subsetting provider frame
        provider_idx = np.concatenate(label).flatten().tolist()
        provider_frame = self.providers.loc[provider_idx]

        #Expanding grantee frame by duplicating values
        g_idx = grantees.index
        expanded_indices = np.repeat(g_idx, k_neighbors)
        grantee_expanded = grantees.loc[expanded_indices]

        #Misspelled provider
        return provider_frame, grantee_expanded, provider_idx
    
    
    def predict(self, k_neighbors: int, grants):
        """This function makes predictions of matches on the batched grantee values"""
        provider_frame, grantee_expanded, provder_idx = self.get_index_labels(k_neighbors, grants)

        #Creating features from both dataframes

        features = self.features_creator.create_distance_features(grantee_expanded, provider_frame)

        #Making predictions and turning into a new column
        pred = self.model.predict(features, proba=True)
        grantee_expanded['pred_proba'] = pd.Series(pred.flatten()) 

        max_indexes = grantee_expanded.groupby('og_grantee_index')['pred_proba'].idxmax()

        matches = pd.DataFrame(columns=['Grantee_ID', 'Provider_ID', 'Is_Match', 'Match_Prob'])

        for idx, i in enumerate(max_indexes):
            pred_value = pred[i]

            if pred_value < 0.5:
                is_match = 0
            else:
                is_match = 1

            row_data = {'Grantee_ID': idx, 'Provider_ID': i, 'Is_Match': is_match, 'Match_Prob': pred_value}
            matches = pd.concat([matches, pd.DataFrame([row_data])], ignore_index=True)

        return matches

if __name__ == '__main__':
 
    #Reading in data
    df_providers = npi.NPIReader('npi_grants/data/npidata_pfile_20240205-20240211.csv').df
    df_grantees = grants.GrantsReader('npi_grants\data\RePORTER_PRJ_C_FY2022.csv').df

    #Getting hnsw index
    hnsw = hnsw.HNSW(df_providers)
    predictor = Predict(df_providers, entity_resolution_model.EnttityResolver(r'npi_grants\data'),
                        '20240403_entity_resolution_model.json', hnsw, 100
                        )
    
    matches = predictor.predict(10, df_grantees)
    matches

    



    
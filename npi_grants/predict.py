import numpy as np
import pandas as pd
import hnswlib 
import fasttext
from npi_grants import db, create_features, entity_resolution_model
from sklearn.neighbors import NearestNeighbors
from npi_grants.data_readers import npi, grants

from npi_grants import hnsw


class Predict:

    def __init__(self, providers: pd.DataFrame, grantees: pd.DataFrame, classifier, classifier_load_path,
                 hnsw_instance, batch_size, embed_model=fasttext.load_model('npi_grants/data/cc.en.50.bin')):
        
        self.providers = providers
        self.grantees = grantees
        self.hnsw_instance = hnsw_instance
        self.embed_model = embed_model
        self.batch_size = batch_size

        classifier.load(classifier_load_path)
        self.model = classifier

    def embed_grantees(self):
        """This function embeds the grantee dataframe fullname"""

        #Calculating a fullname columns
        self.grantees['fullname'] = (self.grantees['forename'].apply(lambda x: x.lower()) + " " + 
        self.grantees['last_name'].apply(lambda x: x.lower()))

         #Embedding the fullname using fasttext
        self.grantees['vector_g'] = self.grantees['fullname'].apply(
            lambda x: self.embed_model.get_sentence_vector(x) if not pd.isnull(x) else None)


    def get_index_labels(self, k_neighbors: int):
        """This function gets the neighbors for each grantee and returns two dataframes for comparison"""

        #Embedding the grantee fullname
        self.embed_grantees()

        #Getting the HNSW (provider) indices
        index =  self.hnsw_instance.get_indices()

        #Chunking grantee dataframe
        self.grantees = self.grantees.iloc[:self.batch_size, :]
        grantee_array = np.array(self.grantees['vector_g'].dropna().tolist())

        label = []

        #Performing KNN queries
        for i in self.grantees.index:
            labels, distances = index.knn_query(grantee_array[i], k=k_neighbors)
            label.append(labels)
        
        #Getting list of indices and subsetting provider frame
        provder_idx = np.concatenate(label).flatten().tolist()
        provider_frame = self.providers.loc[provder_idx]

        #Expanding grantee frame by duplicating values
        g_idx = self.grantees.index
        expanded_indices = np.repeat(g_idx, k_neighbors)
        grantee_expanded = self.grantees.loc[expanded_indices]

        return provider_frame, grantee_expanded, provder_idx
    
    
    def predict(self):
        """This function makes predictions of matches on the batched grantee values"""
        provider_frame, grantee_expanded, provder_idx = self.get_index_labels(10)

        #Creating features from both dataframes
        features = create_features.CreateFeatures.create_distance_features(self, grantee_expanded, provider_frame)

        pred = self.model.predict(features, proba=True)
        grantee_expanded['pred_proba'] = pd.Series(pred.flatten()) 

        grantee_expanded.groupby('og_grantee_index')['pred_proba']

        matches = {}
        for i, prediction_value in enumerate(pred):
 
            grantee_index = grantee_expanded.iloc[i]['og_grantee_index']
            provider_index = provder_idx[i]
        
            key = str(grantee_index) + " " + str(provider_index)

            matches[key] = prediction_value

        return matches

if __name__ == '__main__':
 
    #Reading in data
    df_providers = npi.NPIReader('npi_grants/data/npidata_pfile_20240205-20240211.csv').df
    df_grantees = grants.GrantsReader('npi_grants\data\RePORTER_PRJ_C_FY2022.csv').df

    hnsw = hnsw.HNSW(df_providers)
    index = hnsw.get_indices(space = 'l2', ef_construction=200, M=16)


    predictor = Predict(df_providers, df_grantees, entity_resolution_model.EnttityResolver(r'npi_grants\data'),
                        '20240403_entity_resolution_model.json', hnsw, 100
                        )
    predictor.predict()

    



    
import numpy as np
import pandas as pd
import hnswlib 
import fasttext
from npi_grants import db, create_features, entity_resolution_model
from sklearn.neighbors import NearestNeighbors
from npi_grants.data_readers import npi, grants


#Loading reusable classifier 
model = entity_resolution_model.EnttityResolver(r'npi_grants\data')

#Loading in the trained model, trained using 'train_distance_classifier' script
model.load('20240403_entity_resolution_model.json')

class HNSW:

    def __init__(self, providers: pd.DataFrame, model=fasttext.load_model('npi_grants/data/cc.en.50.bin')):
        """Initializing HNSW with DataFrames"""

        self.providers = providers
        self.model = model

    def embed_name(self):
        """This function creates a full-name column and embeds it using fasttext"""

        #Calculating a fullname columns
        self.providers['fullname'] = (self.providers['forename'].apply(lambda x: x.lower()) + " " + 
        self.providers['last_name'].apply(lambda x: x.lower()))

         #Embedding the fullname using fasttext
        self.providers['vector_p'] = self.providers['fullname'].apply(
            lambda x: self.model.get_sentence_vector(x) if not pd.isnull(x) else None)


    def get_indices(self, space = 'l2', ef_construction=200, M=16):
        """This function gets the HNSW indices"""

        self.embed_name()

        #HNSW Indexing on the provider dataset
        dim = len(self.providers['vector_p'].iloc[0])  
        p = hnswlib.Index(space=space, dim=dim)  
        p.init_index(max_elements=len(self.providers['vector_p']), ef_construction=ef_construction, M=M)

        #Turn dataframe column of arrays into an array of lists
        providers_array =  np.array(self.providers['vector_p'].dropna().tolist())
        ids = self.providers.index.tolist()

        #Add FastText vectors to the index
        p.add_items(providers_array, ids)

        return p
    





import numpy as np
import pandas as pd
import hnswlib 
import fasttext
from npi_grants import db, create_features, entity_resolution_model
from sklearn.neighbors import NearestNeighbors
from npi_grants.data_readers import npi, grants


#Loading reusable classifier 
model = entity_resolution_model.EnttityResolver(r'npi_grants\data')

#Loading fasttext model
ft_model = fasttext.load_model('npi_grants/data/cc.en.50.bin')

#Loading in the trained model, trained using 'train_distance_classifier' script
model.load('20240403_entity_resolution_model.json')

#BLOCK ON NEAREST NEIGHBOR SEARCH
def get_hnsw_indices():
    """This Function returns a lists of HNSW indices for the grantees that are close in nearest neighbors search"""

    #Embedding name columns using fastetext
    df_providers = npi.NPIReader('npi_grants/data/npidata_pfile_20240205-20240211.csv').df
    df_grantees = grants.GrantsReader('npi_grants\data\RePORTER_PRJ_C_FY2022.csv').df

    #Calculating a fullname columns
    df_providers['fullname'] = df_providers['forename'].apply(lambda x: x.lower()) + " " + df_providers['last_name'].apply(lambda x: x.lower())
    df_grantees['fullname'] = df_grantees['forename'].apply(lambda x: x.lower()) + " " + df_grantees['last_name'].apply(lambda x: x.lower())

    #Embedding the fullname using fasttext
    df_providers['vector_p'] = df_providers['fullname'].apply(lambda x: ft_model.get_sentence_vector(x) if not pd.isnull(x) else None)
    df_grantees['vector_g'] = df_grantees['fullname'].apply(lambda x: ft_model.get_sentence_vector(x))

    #HNSW Indexing
    dim = len(df_providers['vector_p'].iloc[0])  
    p = hnswlib.Index(space='cosine', dim=dim)  
    p.init_index(max_elements=len(df_providers['vector_p']), ef_construction=200, M=16)

    #Turn dataframe column of arrays into an array of lists
    providers_array =  np.array(df_providers['vector_p'].dropna().tolist())
    ids = np.arange(len(providers_array))

    #Add FastText vectors to the index
    p.add_items(providers_array, ids)
    
    #Turning the grantees vectors into an array of lists to query from
    grantee_array = np.array(df_grantees['vector_g'].dropna().tolist())

    labels, distances = p.knn_query(grantee_array, k=10)

    return labels


def predict_matches():

    df_providers = npi.NPIReader('npi_grants/data/npidata_pfile_20240205-20240211.csv').df
    df_grantees = grants.GrantsReader('npi_grants\data\RePORTER_PRJ_C_FY2022.csv').df

    feature_extractor = create_features.CreateFeatures()

    for i in range(100):
    
        features = feature_extractor.create_distance_features(df_grantees[i], df_providers.loc[get_hnsw_indices()[i]])
        preds = model.predict(features)

    return (preds==1)



if __name__ == '__main__':
    list = get_potential_matches()
    print("1")



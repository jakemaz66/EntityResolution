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
    """This Function returns a lists of HNSW indices from the providers for each grantee row that are close in a nearest neighbors search"""

    #Reading in data
    df_providers = npi.NPIReader('npi_grants/data/npidata_pfile_20240205-20240211.csv').df
    df_grantees = grants.GrantsReader('npi_grants\data\RePORTER_PRJ_C_FY2022.csv').df

    #Calculating a fullname columns
    df_providers['fullname'] = df_providers['forename'].apply(lambda x: x.lower()) + " " + df_providers['last_name'].apply(lambda x: x.lower())
    df_grantees['fullname'] = df_grantees['forename'].apply(lambda x: x.lower()) + " " + df_grantees['last_name'].apply(lambda x: x.lower())

    #Embedding the fullname using fasttext
    df_providers['vector_p'] = df_providers['fullname'].apply(lambda x: ft_model.get_sentence_vector(x) if not pd.isnull(x) else None)
    df_grantees['vector_g'] = df_grantees['fullname'].apply(lambda x: ft_model.get_sentence_vector(x))

    #HNSW Indexing on the provider dataset
    dim = len(df_providers['vector_p'].iloc[0])  
    p = hnswlib.Index(space='cosine', dim=dim)  
    p.init_index(max_elements=len(df_providers['vector_p']), ef_construction=200, M=16)

    #Turn dataframe column of arrays into an array of lists
    providers_array =  np.array(df_providers['vector_p'].dropna().tolist())
    ids = df_providers.index.tolist()

    #Add FastText vectors to the index
    p.add_items(providers_array, ids)
    
    #Turning the grantees vectors into an array of lists to query from
    grantee_array = np.array(df_grantees['vector_g'].dropna().tolist())

    #Querying the 100 closest provider rows for each grantee
    labels, distances = p.knn_query(grantee_array, k=100)

    return labels


def predict_matches():
    """This function uses the queries HNSW indices to predict matches"""

    #Reading in the data
    df_providers = npi.NPIReader('npi_grants/data/npidata_pfile_20240205-20240211.csv').df
    df_grantees = grants.GrantsReader('npi_grants\data\RePORTER_PRJ_C_FY2022.csv').df

    #Restting index
    # df_grantees = df_grantees.reset_index()
    # df_providers = df_providers.reset_index()

    #Get all the indicies of providers I want to test each grantee on
    feature_extractor = create_features.CreateFeatures()
    indices = get_hnsw_indices()

    #Initializing dictionary to store predictions (matches or non matches) along with grantee index
    matches = {}

    for index, row in df_grantees.iterrows():
        
        #Transforming grantee row to a dataframe
        row = row.to_frame().transpose()

        #Getting the 100 nearest neighbors from the providers for the grantee row
        neighbors = df_providers.loc[indices[index].tolist()]

        for j in range(len(neighbors)):
            orig_ind = neighbors.index
            neighbors.reset_index(inplace=True)

            #Computing the features of
            features = feature_extractor.create_distance_features(row, 
                                                           neighbors.loc[j].to_frame().transpose())
        
            preds = model.predict(features)

            if preds[0] == 1:
                matches[f'{index} + {orig_ind[j]}'] = 1
            else:
                matches[f'{index} + {orig_ind[j]}'] = 0
                print("i")

    return matches
        

if __name__ == '__main__':
    predict_matches()
    print("1")



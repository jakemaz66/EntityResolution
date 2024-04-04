import numpy as np
import pandas as pd
import hnswlib 
import fasttext
from npi_grants import db, create_features, entity_resolution_model
from sklearn.neighbors import NearestNeighbors


#Loading reusable classifier 
model = entity_resolution_model.EnttityResolver(r'npi_grants\data')

#Loading fasttext model
ft_model = fasttext.load_model('npi_grants/data/cc.en.50.bin')

#Loading in the trained model, trained using 'train_distance_classifier' script
model.load('20240403_entity_resolution_model.json')

#BLOCK ON NEAREST NEIGHBOR SEARCH
def get_potential_matches():
    """Get a set of likely matches between grantee/grant and 
    provider/npi. We will use distances to estimate likely matches."""

    #Getting grantee data from database
    query = f'''SELECT id, forename, last_name, 
                         city, state, country
                FROM grantee
                '''
    grantees = (pd.read_sql(query, db.sql()))

   
   #Getting provider data from database
    query = f'''SELECT id, forename, last_name, 
                        city, state, country
                FROM provider
                '''
    providers = pd.read_sql(query, db.sql())

    #Converting data into features
    feature_extractor = create_features.CreateFeatures()
    features = feature_extractor.create_distance_features(grantees, providers)

    #Embedding name columns using fastetext
    features['vector_p'] = features['fullname_p'].apply(lambda x: ft_model.get_sentence_vector(x) if not pd.isnull(x) else None)
    features['vector_g'] = features['fullname_g'].apply(lambda x: ft_model.get_sentence_vector(x))

    #HNSW Indexing
    dim = len(features['vector_p'].iloc[0])  
    p = hnswlib.Index(space='cosine', dim=dim)  
    p.init_index(max_elements=len(features['vector_p']), ef_construction=200, M=16)

    #Turn dataframe column of arrays into an array of lists
    providers_array =  np.array(features['vector_p'].dropna().tolist())
    ids = np.arange(len(providers_array))

    #Add FastText vectors to the index
    p.add_items(providers_array, ids)

    #Turning the grantees vectors into an array of lists to query from
    grantee_array = np.array(features['vector_g'].dropna().tolist())

    #Getting the nearest neighbor k for each grantee
    labels, distances = p.knn_query(grantee_array, k=10)

    #Dropping non-features from dataset
    features_new = features.drop(['fullname_g', 'fullname_p', 'vector_p', 'vector_g'], axis=1)
    index = labels[0][0]

    predict_df = features_new.loc[index].to_frame().transpose()

    preds = model.predict(predict_df)

    return (preds==1)



if __name__ == '__main__':
    list = get_potential_matches()
    print("1")



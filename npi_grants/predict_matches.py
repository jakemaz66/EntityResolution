import numpy as np
import pandas as pd

from npi_grants import db, create_features, entity_resolution_model

#Loading reusable classifier 
model = entity_resolution_model.EnttityResolver(r'npi_grants\data')

#Loading in the trained model, trained using 'train_distance_classifier' script
model.load('20240403_entity_resolution_model.json')

def get_potential_matches():
    """Get a set of likely matches between grantee/grant and 
    provider/npi. We will use distances to estimate likely matches."""

    

    query = f'''SELECT id, forename, last_name, 
                         city, state, country
                FROM grantee
                '''
    
    grantees = (pd.read_sql(query, db.sql()))

   
    query = f'''SELECT id, forename, last_name, 
                        city, state, country
                FROM provider
                '''
    
    providers = pd.read_sql(query, db.sql())

    comb = grantees.merge(providers, on='last_name')

    feature_extractor = create_features.CreateFeatures()
    features = feature_extractor.create_distance_features(grantees, providers)

    preds = model.predict(features)

    return (preds==1)



if __name__ == '__main__':
    list = get_potential_matches()
    print("1")



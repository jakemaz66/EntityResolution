import pandas as pd
import numpy as np
import jarowinkler 
#from sentence_transformers import SentenceTransformer

class CreateFeatures:
    """This class creates distance features between the datasets"""

    def __init__(self):
        pass

    def create_distance_features(self, grantees: pd.DataFrame, providers: pd.DataFrame):
        #Lowercasing and stripping whitespace
        cols = ['forename', 'city', 'state']
        for i in cols:
            grantees[i] = grantees[i].str.lower().str.strip()
            providers[i] = providers[i].str.lower().str.strip()

        #Features for training data
        comb_df = pd.concat([grantees.add_suffix('_g'), providers.add_suffix('_p')], axis=1)

        #Features for testing data -> leaves only one lastname column
        # comb_df = grantees.add_suffix('_g').merge(providers.add_suffix('_p'), 
        #                       how='outer',
        #                       left_on='last_name_g',
        #                       right_on='last_name_p')
        

        #Creating the distance features
        comb_df['jw_dist_forename'] = comb_df.apply(lambda row: jw_dist(row['forename_g'],
                                                                  row['forename_p']), 
                                                                  axis=1)  
        comb_df['set_dist_forename'] = comb_df.apply(lambda row: set_dist(row['forename_g'],
                                                                  row['forename_p']), 
                                                                  axis=1) 
        
        comb_df['jw_dist_city'] = comb_df.apply(lambda row: jw_dist(row['city_g'],
                                                                  row['city_p']), 
                                                                  axis=1)  
        comb_df['set_dist_city'] = comb_df.apply(lambda row: set_dist(row['city_g'],
                                                                  row['city_p']), 
                                                                  axis=1) 
        
        comb_df['jw_dist_state'] = comb_df.apply(lambda row: jw_dist(row['state_g'],
                                                                  row['state_p']), 
                                                                  axis=1)  
        comb_df['set_dist_state'] = comb_df.apply(lambda row: set_dist(row['state_g'],
                                                                  row['state_p']), 
                                                                  axis=1)  

def jw_dist(v1: str, v2: str) -> float:

    if isinstance(v1, str) and isinstance(v2, str):

        return jarowinkler.jarowinkler_similarity(v1, v2)
    else:
        return np.nan


def set_dist(v1: str, v2: str) -> float:
    if isinstance(v1, str) and isinstance(v2, str):
        v1 = set(v1.split(' '))
        v2 = set(v2.split(' '))
        return len(v1.intersection(v2))/min(len(v1), len(v2))
    return np.nan
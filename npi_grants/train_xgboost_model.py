import numpy as np
import pandas as pd

from npi_grants import db, create_features, entity_resolution_model


def fix_missing_pid_mistake(training: pd.DataFrame) -> list[int]:
    """We forgot to add an ID column to the training data, so we will
    calculate it here."""
    names = "'" + training['last_name'] + "'"
    names = ', '.join(names)
    query = f'''SELECT id AS p_id, last_name, forename AS p_forename, city AS p_city, 
                state AS p_state, country AS p_country
                FROM provider
                WHERE last_name IN ({names})'''
    providers = pd.read_sql(query, db.sql())

    training_length = len(training)
    training = training.merge(providers, on=['last_name',
                                             'p_forename',
                                             'p_city',
                                             'p_state',
                                             'p_country'])
    assert len(training) == training_length
    return training


def create_training_data_features_labels():
    """Load known training data, extract values from db, convert to 
    features."""

    training = pd.read_csv('npi_grants/data/likely_grantee_provider_matches.csv')
    #training = fix_missing_pid_mistake(training)
    grantee_ids = ', '.join([str(v) for v in training['g_id']]) 

    #Get all the values from the likely ids
    query = f'''SELECT id AS g_id,
                       TRIM(LOWER(last_name)) AS last_name, 
                       TRIM(LOWER(forename)) AS forename, 
                       TRIM(LOWER(city)) AS city, 
                       TRIM(LOWER(state)) AS state
                FROM grantee
                WHERE id IN ({grantee_ids})'''
    
    grantees = pd.read_sql(query, db.sql())


    grantees = training.merge(grantees, how='left', on='g_id')[
        ['g_forename', 'g_city', 'g_state']].rename(columns={
            'g_forename': 'forename',
            'g_state': 'state',
            'g_city': 'city'
        })

    #Doing same thing for providers
    provider_ids = ', '.join([str(v) for v in training['p_id']]) 

    query = f'''SELECT id AS p_id,
                       TRIM(LOWER(last_name)) AS last_name, 
                       TRIM(LOWER(forename)) AS forename, 
                       TRIM(LOWER(city)) AS city, 
                       TRIM(LOWER(state)) AS state
                FROM provider
                WHERE id IN ({provider_ids})'''
    
    providers = pd.read_sql(query, db.sql())

    providers = training.merge(providers, how='left', on='p_id')[
        ['p_forename', 'p_city', 'p_state']].rename(columns={
            'p_forename': 'forename',
            'p_state': 'state',
            'p_city': 'city'
        })

    assert len(providers) == len(grantees)
    feature_extractor = create_features.CreateFeatures()
    features = feature_extractor.create_distance_features(grantees, providers)
    labels = training['is_match'].values
    return features, labels


def create_xgb_model(features: pd.DataFrame, labels: np.ndarray, path: str):
    xgb_model = entity_resolution_model.EnttityResolver(r'npi_grants\data')
    xgb_model.train(features, labels)
    xgb_model.save(path)


if __name__ == '__main__':
    features, labels = create_training_data_features_labels()
    create_xgb_model(features, labels, 'entity_resolution_model.json')
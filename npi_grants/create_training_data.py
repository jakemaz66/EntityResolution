import pandas as pd
from npi_grants import db, create_features


def sample_last_names():
    """Get a sample of last names from both databases
    """
    df = pd.read_sql('''SELECT DISTINCT gr.last_name
                        FROM grantee gr
                        INNER JOIN provider pr
                            ON gr.last_name = pr.last_name
                        LIMIT 100;''', db.sql())
    df = df.loc[~df['last_name'].str.contains("'")]
    return df


def get_probable_matches():
    """Get a set of likely matches between grantee/grant and 
    provider/npi. We will use distances to estimate likely matches."""

    sample = sample_last_names()

    sample['last_name'] = "'" + sample['last_name'] + "'"
    names = ', '.join(sample['last_name'])

    query = f'''SELECT id, forename, last_name, 
                        organization, city, state, country
                FROM grantee
                WHERE last_name IN ({names})'''
    
    grantees = (pd.read_sql(query, db.sql())
                   .add_prefix('g_')
                   .rename(columns={'g_last_name': 'last_name'}))

   
    query = f'''SELECT id AS p_id, last_name, forename AS p_forename, city AS p_city, 
                state AS p_state, country AS p_country
                FROM provider
                WHERE last_name IN ({names})'''
    
    providers = pd.read_sql(query, db.sql())

    comb = grantees.merge(providers, on='last_name')

    comb['forename_jw_dist'] = comb.apply(
        lambda row: create_features.jw_dist(row['g_forename'], row['p_forename']),
          axis=1)
    
    return (comb.sort_values(by='forename_jw_dist', ascending=False)
                .groupby('last_name')
                .head(5))


if __name__ == '__main__':
    matches = get_probable_matches()
    matches.to_csv('npi_grants/data/likely_grantee_provider_matches.csv', index=False)
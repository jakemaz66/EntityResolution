import pandas as pd
from npi_grants import db


def sample_last_names():
    """Get a sample of last names from both databases
    """
    df = pd.read_sql('''SELECT DISTINCT gr.last_name
                        FROM grantee gr
                        INNER JOIN provider pr
                            ON gr.last_name = pr.last_name
                        LIMIT 100;''', db.sql())
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
    
    grantees = pd.read_sql(query, db.sql())


if __name__ == '__main__':
    get_probable_matches()
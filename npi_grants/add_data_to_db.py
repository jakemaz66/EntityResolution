import sqlalchemy
from data_readers import grants, npi
import db


def npi_csv_to_db(csv_path: str):
    """
    This function creates a relational database out of the npi csv dataset
    """

    #Make npi data, have to rename columns, match data types, define bridge table, download beekeeper studio
    df = npi.NPIReader(csv_path).df

    #Subsetting to desired columns
    df = df[['npi', 'taxonomy_code', 'last_name', 'forename', 'address', 'cert_date', 'city', 'state', 'country']]
 

    #Dropping NaNs to enforce NOT NULL parameter
    df.dropna(inplace=True)

    #Translating pandas dataframe to database
    df.to_sql('provider',
              db.sql(),
              if_exists='append',
              index=False
              #Big Data
              #method = 'multi'
              #chunksize=1000
              )
    
    
def grants_csv_to_db(csv_path: str):
    """
    This function creates a relational database out of the grants csv dataset
    """

    #Reading in data
    df = grants.GrantsReader(csv_path).df

    #Subsetting to desired columns
    df = df[['application_id',
                 'budget_start',
                 'grant_type',
                 'total_cost',
                 'organization',
                 'city',
                 'state',
                 'country',
                 'forename',
                 'last_name',
                 'is_contact']]


    #Dropping NaNs to enforce NOT NULL parameter
    df.dropna(inplace=True)

    #Translating pandas dataframe to database
    df.to_sql('grantee',
              db.sql(),
              if_exists='append',
              index=False
              )
    

if __name__ == '__main__':
    npi_csv_to_db('npi_grants/data/npidata_pfile_20240205-20240211.csv')
    grants_csv_to_db('npi_grants/data/RePORTER_PRJ_C_FY2022.csv')
import pandas as pd

class GrantsReader():
    """This class reads in and returns a cleaned grants dataframe"""

    def __init__(self, path: str):
        """Inittializes the dataframe"""
        self.df = self._read(path)

    def to_db(self, conn):
        """Turns the Pandas Dataframe into a table in the SQL database"""
        self.df.to_sql(
            'grantee',
            conn,
            if_exists='append',
            index=False
        )

    def _read(self, path):
        """Read in the data and rename columns"""
        df = pd.read_csv(path)

        df = df.rename(columns={
            'APPLICATION_ID': 'application_id',
            'BUDGET_START': 'budget_start',
            'ACTIVITY': 'grant_type',
            'TOTAL_COST': 'total_cost',
            'PI_NAMEs': 'pi_names',
            'PI_IDS': 'pi_ids',
            'ORG_NAME': 'organization',
            'ORG_CITY': 'city',
            'ORG_STATE': 'state',
            'ORG_COUNTRY': 'country'
        })
        
        #Calling clean method to handle names
        df = self._clean(df)

        df = df[['application_id',
                 'budget_start',
                 'grant_type',
                 'total_cost',
                 'organization',
                 'city',
                 'state',
                 'country',
                 'forename',
                 'lastname',
                 'is_contact']]

        return df


    def _clean(self, df: pd.DataFrame):
        """Clean the names columns and returning new names"""
        df['pi_names'] = df['pi_names'].str.split(';')
        df = df.explode('pi_names')

        df['is_contact'] = df['pi_names'].str.lower().str.contains('(contact)', regex=False)
        df['pi_names'] = df['pi_names'].str.replace('(contact)', '')

        df['both_names'] = df['pi_names'].apply(lambda x: x.split(',')[:2])
        df['forename'] = df['both_names'].apply(lambda x: x[0])
        df['lastname'] = df['both_names'].apply(lambda x: x[1])

        return df

if __name__ == '__main__':
    reader = GrantsReader('npi_grants/data/RePORTER_PRJ_C_FY2022.csv')
    print(reader.df)


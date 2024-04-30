import pandas as pd


class NPIReader():
    def __init__(self, path: str):
        """Pass in path to NIH grants file to clean and read"""
        self.df = self._read(path)

    def to_db(self, con):
        """Send the data to a database via a SQLAlchemy connection"""
        self.df.to_sql('provider',
                       con,
                       if_exists='append',
                       index=False)

    def _read(self, path: str):
        """Read in a grants NIH file and return a clean dataframe"""
        df = pd.read_csv(path)
        df = df.rename(columns={
            'NPI': 'npi',
            'Healthcare Provider Taxonomy Code_1': 'taxonomy_code',
            'Provider Last Name (Legal Name)': 'last_name',
            'Provider First Name': 'forename',
            'Provider First Line Business Practice Location Address': 'address',
            'Certification Date': 'cert_date',
            'Provider Business Practice Location Address City Name': 'city',
            'Provider Business Practice Location Address State Name': 'state',
            'Provider Business Practice Location Address Country Code (If outside U.S.)': 'country'
 
        })
        df = df[['npi',
                 'taxonomy_code',
                 'last_name',
                 'forename',
                 'address',
                 'cert_date',
                 'city',
                 'state',
                 'country']]
        df = df.dropna(subset=['last_name'])
        return df
    

if __name__ == '__main__':
    reader = NPIReader('npi_grants/data/npidata_pfile_20240205-20240211.csv')
    print(reader.df)
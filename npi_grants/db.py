import sqlalchemy
import sqlite3


def create_db():
    """A database creation statement"""
    query = '''
    CREATE TABLE IF NOT EXISTS npi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lastname VARCHAR(100) NOT NULL,
        forename VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    '''

    conn = sqlite3.connect('data/grant_npi.db')
    cursor = conn.cursor()
    cursor.execute(query)
    cursor.close()


def sql():
    """Return a connection to our SQL database"""
    engine = sqlalchemy.create_engine('sqlite:///npi_grants/data/grant_npi.db')
    return engine.connect()
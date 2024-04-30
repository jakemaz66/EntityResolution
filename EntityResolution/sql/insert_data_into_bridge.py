import sqlite3

#Connecting to Database
conn = sqlite3.connect('npi_grants/data/grant_npi.db')
cursor = conn.cursor()

#Defining query
sql_query = """
    INSERT INTO grantee_provider (grantee_id, provider_id)
    SELECT npi.id AS npi_id, grants.id AS grants_id
    FROM npi 
    JOIN grants 
    ON LOWER(npi.lastname) = LOWER(grants.lastname)
    WHERE grants.forename IS NOT NULL
    AND grants.city IS NOT NULL
    LIMIT 50000;
"""

#Executing Queries and closing connection
cursor.execute(sql_query)
conn.commit()

conn.close()
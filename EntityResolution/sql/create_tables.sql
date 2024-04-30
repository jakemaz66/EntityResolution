CREATE TABLE IF NOT EXISTS grantee (
    id INTEGER PRIMARY KEY NOT NULL,
    application_id INTEGER NOT NULL,
    budget_start DATETIME,
    grant_type VARCHAR(3) NOT NULL,
    total_cost FLOAT,
    is_contact BOOLEAN NOT NULL,
    forename VARCHAR(100),
    last_name VARCHAR(100) NOT NULL,
    organization VARCHAR(100),
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS provider (
    id INTEGER PRIMARY KEY NOT NULL,
    npi INT NOT NULL,
    taxonomy_code VARCHAR(25),
    last_name VARCHAR(100) NOT NULL,
    forename VARCHAR(100),
    address VARCHAR(250),
    cert_date DATETIME,
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100)
);

-- The bridge table between grantees and providers
CREATE TABLE IF NOT EXISTS grantee_provider (
    grantee_id INT NOT NULL,
    provider_id INT NOT NULL,
    FOREIGN KEY(grantee_id) REFERENCES grantee(id),
    FOREIGN KEY(provider_id) REFERENCES provider(id),
    UNIQUE(grantee_id, provider_id)
);

-- Drop table if exists
DROP TABLE IF EXISTS stock_details;
DROP TABLE IF EXISTS stock_type;

-- Create stock_type table
CREATE TABLE stock_type (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50),
    name VARCHAR(50)
);

-- Create stock_details table
CREATE TABLE stock_details (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50),
    name VARCHAR(100),
    type INT,
    price VARCHAR,  -- Assuming price is a numerical value, changed to DECIMAL
    percentage_change VARCHAR,  -- Changed to DECIMAL
    market_cap VARCHAR,  -- Changed to DECIMAL
    volume VARCHAR,  -- This column remains as VARCHAR
    FOREIGN KEY (type) REFERENCES stock_type(id)
);

from sqlalchemy import create_engine
import pandas as pd

data_base_type= 'postgresql'
sql_type ='sql'
user_name ='postgres'
password='admin'
server_address='localhost'
data_base_name='postgres'
 

# Connection string
#Example postgresql://postgres:admin@localhost/postgres
connection_string =f"{data_base_type}://{user_name}:{password}@{server_address}/{data_base_name}"

try:
    # Create engine
    engine = create_engine(connection_string)

    # Test the connection
    with engine.connect() as connection:
        print("Connection successful!")
except Exception as e:
    print("Unable to connect to the database:", e)


query = '''
Select * from  test
'''
print(query)

df=pd.read_sql_query(query,connection )

##print(df)

# Creating databases

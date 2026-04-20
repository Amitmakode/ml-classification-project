# This file loads transformed data into a new MySQL table for ML use

import pandas as pd  # data manipulation library
from sqlalchemy import create_engine  # database connection engine
from dotenv import load_dotenv  # load environment variables
import os  # access environment variables

load_dotenv()  # load .env file

def get_engine():
    # build mysql connection string using env variables
    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)  # return sqlalchemy engine object

def load_data(df):
    engine = get_engine()  # get database connection
    # write dataframe to new table 'transformed_shoppers'
    # if_exists='replace' means drop and recreate table every time
    df.to_sql('transformed_shoppers', engine, if_exists='replace', index=False)
    print(f"Loaded {len(df)} rows into transformed_shoppers table")  # log success

if __name__ == "__main__":
    from extract import extract_data  # import extract function
    from transform import transform_data  # import transform function
    df = extract_data()  # fetch raw data from MySQL
    df = transform_data(df)  # clean and encode data
    load_data(df)  # load into transformed_shoppers table
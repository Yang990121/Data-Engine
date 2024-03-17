import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

Base = declarative_base()


class DatabaseManager:
    def __init__(self, database_name="is3107"):
        load_dotenv()

        self.pg_user = os.getenv("PG_USER")
        self.pg_password = os.getenv("PG_PASSWORD")
        self.pg_db = database_name

        self.db_uri = f"postgresql://{self.pg_user}:{self.pg_password}@localhost:5432/{self.pg_db}"
        self.engine = create_engine(self.db_uri)
        self.Session = sessionmaker(bind=self.engine)

    def create_table_from_df(self, df, table_name='housing_data'):
        column_types = {
            'object': String,
            'int64': Integer,
            'float64': Float,
            'datetime64[ns]': DateTime,
        }

        class DynamicTable(Base):
            __tablename__ = table_name

            # Dynamically add columns
            for col, dtype in zip(df.columns, df.dtypes):
                if col == df.columns[0]:
                    exec(f"{col} = Column(Integer, primary_key=True)")
                else:
                    sql_type = column_types.get(str(dtype), String)
                    exec(f"{col} = Column(sql_type)")

        # Create the table in the database if it doesn't exist
        Base.metadata.create_all(self.engine)

    def load_df_to_db(self, df, table_name='housing_data'):
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)

    def read_table(self, table_name='housing_data'):
        """
        By right we should give more parameters to give a more complex SQL function
            Because this will be faster to pull
        But that is troublesome, since this is a small project, we just pull the whole db and manipulate using pandas LOL
        :param table_name:
        :return:
        """
        query = f"SELECT * FROM {table_name};"
        return pd.read_sql_query(query, self.engine)

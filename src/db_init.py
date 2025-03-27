import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Boolean, Integer, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

# Define a base class for declarative models
Base = declarative_base()

class InferenceResult(Base):
    """
    Table to store the inference results.
    """
    __tablename__ = "inference_results"
    pid = Column(String, primary_key=True)           # property id
    vid = Column(String)                               # if needed for reference
    solar_detection_result = Column(Boolean)           # True/False detection result
    confidence = Column(Float)                         # confidence score
    city_code = Column(String)                         # e.g., "GM0153"
    __table_args__ = (UniqueConstraint("pid", name="uix_pid"),)

def get_engine():
    """
    Creates and returns a SQLAlchemy engine for Postgres.
    Adjust the connection string as needed.
    """
    # Replace with your Postgres credentials and database
    user = "postgres"
    password = "password"
    host = "localhost"
    port = "5432"
    database = "mydb"
    connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(connection_string, echo=True)
    return engine

def init_db():
    """
    Initializes the database by creating the required tables.
    This creates the inference_results table (and any other defined models).
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Database initialized with tables: inference_results")
    return engine

def import_properties_csv(csv_path, table_name="properties"):
    """
    Imports the energy data CSV into a Postgres table.
    
    The CSV is assumed to be semicolon-delimited.
    The 'identificatie' column is renamed to 'pid'.
    
    :param csv_path: Path to the energy data CSV file.
    :param table_name: Name of the table to create (default is "properties").
    """
    # Read the CSV with semicolon delimiter
    df = pd.read_csv(csv_path, sep=";")
    # Rename 'identificatie' to 'pid'
    df.rename(columns={"identificatie": "pid"}, inplace=True)
    
    # Create a SQLAlchemy engine and import the DataFrame using to_sql
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Properties data imported into table '{table_name}'.")

if __name__ == "__main__":
    # Initialize the database (creates the inference_results table)
    engine = init_db()
    
    # Import the energy CSV into the 'properties' table.
    # Adjust the CSV path to your raw data file.
    properties_csv_path = "../data/raw/energy_data_enschede.csv"
    import_properties_csv(properties_csv_path, table_name="properties")

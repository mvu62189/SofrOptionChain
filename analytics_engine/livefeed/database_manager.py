# database_manager.py
"""
Handles the database connection and table creation.
"""

from sqlmodel import SQLModel, create_engine
import config

print(f"--- LOADING CONFIG FROM: {config.__file__} ---")
# Get the database filename from the config
sqlite_file_name = config.DATABASE_FILE
sqlite_url = f"sqlite:///{sqlite_file_name}"

# The 'engine' is the single point of contact for our database.
# 'echo=True' will print all SQL statements to the console,
# which is extremely useful for debugging.
engine = create_engine(sqlite_url, echo=True)


def create_db_and_tables():
    """
    Creates the database file (if it doesn't exist)
    and all tables defined by our SQLModels.
    """
    print("Creating database and tables...")
    # This command finds all classes that inherit from SQLModel
    # and have 'table=True' and creates them in the database.
    SQLModel.metadata.create_all(engine)
    print("Database and tables created successfully.")

if __name__ == "__main__":
    # This allows us to run 'python database_manager.py'
    # one time to initialize the database.
    create_db_and_tables()
from .database_manager import DatabaseManager

# Create instance only when needed
def get_db_manager():
    return DatabaseManager() 
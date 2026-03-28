# database/init_db.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import init_db

if __name__ == '__main__':
    init_db()
    print("All tables created successfully.")
    print("Database file: database/dr_records.db")
import sqlite3
import os

# Connect to the database
db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "solves.db")
conn = sqlite3.connect(db_path)

query_no_fix = "SELECT COUNT(*) FROM solves WHERE reconstruction NOT LIKE '%fix%'"
cursor = conn.cursor()
cursor.execute(query_no_fix)
result = cursor.fetchone()
print(f"Number of solves with no fixes: {result[0]}")

query_total = "SELECT COUNT(*) FROM solves"
cursor.execute(query_total)
result = cursor.fetchone()
print(f"Total number of solves: {result[0]}")

# Get max and min and average time of solves with no fixes
query_min_time = "SELECT MIN(time) FROM solves WHERE reconstruction NOT LIKE '%fix%'"
cursor.execute(query_min_time)
result = cursor.fetchone()
print(f"Minimum time of solves with no fixes: {result[0]}")

query_max_time = "SELECT MAX(time) FROM solves WHERE reconstruction NOT LIKE '%fix%'"
cursor.execute(query_max_time)
result = cursor.fetchone()
print(f"Maximum time of solves with no fixes: {result[0]}")

query_avg_time = "SELECT AVG(time) FROM solves WHERE reconstruction NOT LIKE '%fix%'"
cursor.execute(query_avg_time)
result = cursor.fetchone()
print(f"Average time of solves with no fixes: {result[0]}")

conn.close()
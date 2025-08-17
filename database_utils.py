import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'solves.db')
conn = None

def get_db_connection():
    global conn
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    return conn

def close_db_connection():
    global conn
    if conn is not None:
        conn.close()
        conn = None

def get_solves_with_no_fix(solve_id: bool = True, solver: bool = True, event: bool = True, method: bool = True, time: bool = True, scramble: bool = True, reconstruction: bool = True):
    columns = []
    if solve_id:
        columns.append("solve_id")
    if solver:
        columns.append("solver")
    if event:
        columns.append("event")
    if method:
        columns.append("method")
    if time:
        columns.append("time")
    if scramble:
        columns.append("scramble")
    if reconstruction:
        columns.append("reconstruction")
    
    sql_query = f"SELECT {', '.join(columns)} FROM solves WHERE reconstruction NOT LIKE '%fix%'"
    df = pd.read_sql_query(sql_query, get_db_connection())
    return df

def get_solves_with_no_fix_count():
    sql_query = "SELECT COUNT(*) FROM solves WHERE reconstruction NOT LIKE '%fix%'"
    result = get_db_connection().execute(sql_query).fetchone()
    return result[0]


import requests
from bs4 import BeautifulSoup
import time
import sqlite3
import argparse
import os

def get_solves_from_main_page(url, method, event):
    """
    Fetches the main page and extracts a list of solves, each as a 
    dictionary containing its id, event, and method.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        solve_table = soup.find('table', id='sortableTable')
        
        if not solve_table:
            print("Error: Could not find the main solve table.")
            return []
            
        rows = solve_table.find('tbody').find_all('tr')
        
        solves_list = []
        for row in rows:
            cells = row.find_all('td')
            if len(cells) > 4 and cells[4].text.strip().upper() == method and cells[1].text.strip().upper() == event: # Ensure the row has enough columns
                solve_info = {
                    "id": cells[0].text.strip(),
                    "solver": cells[3].text.strip(),
                    "event": cells[1].text.strip(),
                    "method": cells[4].text.strip(),
                    "time": float(cells[2].text.strip())
                }
                solves_list.append(solve_info)
                
        print(f"Successfully found {len(solves_list)} {method} {event} solves on the main page.")
        return solves_list
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching main page: {e}")
        return []

def scrape_solve_details(solve_id):
    """
    Visits an individual solve page and returns a dictionary with the 
    scramble and reconstruction.
    """
    solve_url = f"https://reco.nz/solve/{solve_id}"
    print(f"Scraping Solve ID: {solve_id}")
    
    try:
        response = requests.get(solve_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        reco_div = soup.find('div', id='reconstruction')
        if not reco_div:
            return None

        # Extract Scramble
        all_text_nodes = reco_div.find_all(string=True, recursive=False)
        scramble = all_text_nodes[0].strip() if all_text_nodes else ""
        
        # Extract Solve Parts
        lines = reco_div.get_text(separator='\n').strip().split('\n')
        solve_parts_list = [part.strip() for part in lines[1:] if part.strip()]
        reconstruction = "\n".join(solve_parts_list)
        
        return {
            "scramble": scramble,
            "reconstruction": reconstruction
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching details for solve {solve_id}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--event", type=str, required=True)
    parser.add_argument("--limit", type=int, required=False, default=None)
    parser.add_argument("--db_file", type=str, required=False, default="solves.db")
    args = parser.parse_args()

    DESIRED_METHOD = args.method.upper()
    DESIRED_EVENT = args.event.upper()
    LIMIT = args.limit

    print(f"Starting scraper for {DESIRED_EVENT} solves using the {DESIRED_METHOD} method...")

    # 1. Get the list of all solves with their basic info
    all_solves_info = get_solves_from_main_page("https://reco.nz/", DESIRED_METHOD, DESIRED_EVENT)

    # 2. Setup and connect to the SQLite Database
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", args.db_file)
    print(f"\nSaving data to {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 3. Create the table with the new columns
    cur.execute("""
    CREATE TABLE IF NOT EXISTS solves (
        solve_id INTEGER PRIMARY KEY,
        solver TEXT,
        event TEXT,
        method TEXT,
        time REAL,
        scramble TEXT,
        reconstruction TEXT
    )
    """)

    # 4. Loop through solves, get details, and insert into the database
    for solve_info in (all_solves_info if LIMIT is None else all_solves_info[:LIMIT]):
        details = scrape_solve_details(solve_info['id'])
        
        if details:
            # Insert all the data into the new table structure
            cur.execute("""
                INSERT OR IGNORE INTO solves (solve_id, solver, event, method, time, scramble, reconstruction) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(solve_info['id']),
                solve_info['solver'],
                solve_info['event'],
                solve_info['method'],
                solve_info['time'],
                details['scramble'],
                details['reconstruction']
            ))
            conn.commit()
        
        time.sleep(0.7) # Be polite!

    # Close the database connection
    conn.close()

    print("\n--- Scraping complete! Data saved to solves.db ---")
import sqlite3
conn = sqlite3.connect('logs.db')
for row in conn.execute("SELECT * FROM logs"):
    print(row)
conn.close()
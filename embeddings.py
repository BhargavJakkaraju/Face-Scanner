import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os
from huggingface_hub import hf_hub_url, hf_hub_download as cached_download

# Connect to PostgreSQL
conn = psycopg2.connect(
    "postgres://avnadmin:***REMOVED***@pg-1f8db37e-bhargavjakkaraju-1270.d.aivencloud.com:26527/defaultdb?sslmode=require"
)
cur = conn.cursor()

# Create table if it doesn't exist
cur.execute("""
CREATE TABLE IF NOT EXISTS pic (
    filename TEXT PRIMARY KEY,
    embedding FLOAT8[]
)
""")
conn.commit()

ibed = imgbeddings()


for filename in os.listdir("stored-faces"):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue 
    img_path = os.path.join("stored-faces", filename)
    img = Image.open(img_path)
    embedding = ibed.to_embeddings(img)
    cur.execute("INSERT INTO pic (filename, embedding) VALUES (%s, %s) ON CONFLICT (filename) DO NOTHING",
                (filename, embedding[0].tolist()))
    print(f"Inserted {filename}")

conn.commit()
cur.close()
conn.close()

# Add Embeddings + Search.
# Step 2: Convert document into embeddings and perform similarity search

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
import os
from openai import AzureOpenAI

# Initialize client.
print(f"Initialize client...")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# TODO: Load the document "company_policy.txt".
# Use "utf-8" encoding to read the file.
# Read the contents of the file into a variable named "document".
print(f"Load document...")
___


# Split into simple chunks (basic chunking for demo)
print(f"Chunking...")
chunks = document.split("\n")

# Create embeddings for each chunk.
print(f"Embedding...")
embeddings = []
for chunk in chunks:
    # TODO: For each chunk, create an embedding usinfg the Azure OpenAI client object.
    # Skip empoty lines.
    if chunk.strip():  # skip empty lines
        # TODO: Use the client.embeddings.create() method passing it:
        #  - model: the embedding model name from env var AZURE_OPENAI_EMBEDDING_DEPLOYMENT.
        #  - input: the chunk text extracted from the document.
        emb = ___

        # TODO: Append the chunk and its embedding to the "embeddings" list as a tuple.
        # Use emb.data[0].embedding to grt the embedding vector from the response.
        ___

# Function to calculate similarity (dot product)
def similarity(a, b):
    print(f"Calculate similarity...")
    return sum(x*y for x, y in zip(a, b))

# TODO: Define the User query.
query = "___"

# Embed query.
# TODO: Embed the user query using the same embedding model as before.
# Use client.embeddings.create(), passing it the query as input
# and the embedding model name from the env var AZURE_OPENAI_EMBEDDING_DEPLOYMENT.
print(f"Embed query...")
query_embedding = client.embeddings.create(
    ___
).data[0].embedding

# Find most relevant chunk.
print(f"Find most relevant chunk...")
best_chunk = max(
    embeddings,
    key=lambda x: similarity(query_embedding, x[1])
)

print(f"\n Query: {query}")
print("\n--- Most Relevant Chunk ---")
print(best_chunk[0])

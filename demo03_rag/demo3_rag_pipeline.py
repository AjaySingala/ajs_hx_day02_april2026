# Full RAG (Grounded Answer).
# Step 3: Full RAG → retrieve + augment + generate grounded answer

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path).
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
import os
from openai import AzureOpenAI

# Initialize client.
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


# Chunking.
print(f"\n Chunking...")
chunks = document.split("\n")

# Create embeddings.
print(f"\n Create embeddings...")
embeddings = []
for chunk in chunks:
    if chunk.strip():
        emb = client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            input=chunk
        )
        embeddings.append((chunk, emb.data[0].embedding))

# Similarity function.
def similarity(a, b):
    print(f"\n similarity()...")
    return sum(x*y for x, y in zip(a, b))

# Query
query = "What is the meal allowance for employees?"

# Embed query.
print(f"\n Embed query...")
query_embedding = client.embeddings.create(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    input=query
).data[0].embedding

# Retrieve top relevant chunk.
print(f"\n Retrieve top relevant chunk...")
best_chunk = max(
    embeddings,
    key=lambda x: similarity(query_embedding, x[1])
)[0]

# # Augment prompt with retrieved context.
# TODO: Create an augmented prompt that includes the retrieved chunk as context.
# Instruct the model to answer the quesiton using ONLY the provided context,
# and to say "I don't know" if the answer is not in the context.
# Pass the user query and the retrieved chunk in the prompt.
augmented_prompt = f"""
___
"""

# TODO: Call LLM with grounded context.
# Use the client.embeddings.create() and specify the model and messages parameters.
# For messages:
#   - Define a "system" role message to set the behavior of the assistant.
#   - Define a "user" role message to pass the augemnted prompt as content.
print(f"\n Call LLM with grounded context...")
response = ___

print("\n--- RAG Response (Grounded) ---")
print(response.choices[0].message.content)

# TODO: Define another query that is NOT answerable from the 
# provided context to test the model's ability to say "I donb't know".
query = "___"

# TODO: Call LLM with this new query.
# Use the client.embeddings.create() and specify the model and messages parameters.
# For messages:
#   - Define a "system" role message to set the behavior of the assistant.
#   - Define a "user" role message to pass the augmented prompt as content.
print(f"\n Call LLM with unknown context...")
response = ___

print("\n--- RAG response (unknown context) ---")
print(response.choices[0].message.content)


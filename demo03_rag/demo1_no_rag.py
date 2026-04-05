# Simple LLM call WITHOUT any context → prone to hallucination

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

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# TODO: Initialize Azure OpenAI client
client = ___

# TODO: Get the deployment name from the environment variable.
deployment = ___

# TODO: Ask a question that may NOT be present in the model's provided context.
question = "___"

# Direct LLM call (no grounding)
# TODO: call the Azure OpenAI client to get a response for the question.
# Use the client.chat.completions.create() method.
# Provide values for model, messages.
# Define a "system" role message to set the behavior of the assistant.
# Define a "user" role message to pass the question as content.
response = ___


print("\n--- LLM Response (No RAG) ---")
print(f"Question: {question} \n")
print(response.choices[0].message.content)

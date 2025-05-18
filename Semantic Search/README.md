# Semantic Search

This directory contains Python scripts and resources for performing semantic search on question-answer datasets using embeddings and vector stores. It demonstrates how to process data from CSV files or Supabase tables, generate embeddings with OpenAI, and perform similarity search using Chroma.

## Features

- **CSV-based Semantic Search:**  
  Load questions from a CSV file, generate embeddings, and perform semantic search.

- **Supabase-based Semantic Search:**  
  Fetch questions from a Supabase table, store embeddings back to Supabase, and enable semantic search.

- **Evaluation Function:**  
  Evaluate the accuracy of semantic search results against target questions.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Files

- `semantic_search_in_csv.py`  
  Example of semantic search using a local CSV file.

- `semantic_search_in_supabase.py`  
  Example of semantic search using data from a Supabase table.

- `requirements.txt`  
  List of required Python packages.

## Usage

1. **Set up your environment:**  
   Make sure you have Python 3.8+ and install the requirements.

2. **Configure API Keys:**  
   - For OpenAI embeddings, set your OpenAI API key as an environment variable:  
     ```bash
     export OPENAI_API_KEY=your_openai_api_key
     ```
   - For Supabase, update the `url` and `key` variables in `semantic_search_in_supabase.py`.

## Notes

- Ensure your CSV file or Supabase table contains the necessary fields (e.g., `question`, `answer`).
- The scripts will create a local Chroma vector store in the `./chroma_db_carbon_questions` directory.
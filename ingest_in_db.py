# import basics
import os
from dotenv import load_dotenv
import time
from tqdm import tqdm
import re

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# import supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Function to clean text of problematic characters
def clean_text(text):
    if text is None:
        return ""
    # Remove null bytes and other problematic control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Replace other potentially problematic Unicode characters
    text = text.replace('\\u0000', '')
    return text

# Clean document content
def clean_documents(docs):
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        # Clean metadata values that are strings
        for key, value in doc.metadata.items():
            if isinstance(value, str):
                doc.metadata[key] = clean_text(value)
    return docs

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("🔄 Loading documents from 'documents' folder...")
# load pdf docs from folder 'documents'
loader = PyPDFDirectoryLoader("documents")

# split the documents in multiple chunks
documents = loader.load()
print(f"📚 Loaded {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"✂️ Split into {len(docs)} chunks")

# Clean the documents to remove problematic characters
print("🧹 Cleaning document text to remove problematic characters...")
docs = clean_documents(docs)

print("\n🔍 Processing and uploading document chunks to Supabase...")
# Process chunks in batches to show progress
batch_size = 10
total_batches = (len(docs) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(docs), batch_size), desc="Uploading batches", total=total_batches):
    batch = docs[i:i+batch_size]
    try:
        # store chunks in vector store
        vector_store = SupabaseVectorStore.from_documents(
            batch,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=1000,
        )
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    except Exception as e:
        print(f"\n❌ Error processing batch {i//batch_size + 1}: {str(e)}")
        continue

print("\n✅ Document ingestion complete! All chunks have been uploaded to Supabase.")
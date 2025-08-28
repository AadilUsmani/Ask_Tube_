# app/llm.py
import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

# Load environment variables once
load_dotenv()

# Expected env vars (set these in .env):
#   AZURE_OPENAI_API_KEY          = <your key>          # used by SDK via env
#   AZURE_OPENAI_ENDPOINT         = https://<resource>.openai.azure.com/ (SDK reads from env)
#   AZURE_API_VERSION             = 2024-08-01-preview  (or your version)
#   AZURE_CHAT_DEPLOYMENT         = gpt-4o-mini         (or your chat deployment)
#   AZURE_EMBEDDING_DEPLOYMENT    = text-embedding-3-small
#   LLM_TEMPERATURE               = 0.0 (optional)

@lru_cache(maxsize=1)
def get_answer_llm():
    """Main LLM for Q&A / summarization with adjustable temperature."""
    deployment = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")
    api_version = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
    temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    try:
        # Newer param name
        return AzureChatOpenAI(azure_deployment=deployment, openai_api_version=api_version, temperature=temp)
    except TypeError:
        # Back-compat for some langchain_openai versions
        return AzureChatOpenAI(deployment_name=deployment, api_version=api_version, temperature=temp)

@lru_cache(maxsize=1)
def get_cleaner_llm():
    """Deterministic LLM used for grammar cleanup."""
    deployment = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")
    api_version = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
    try:
        return AzureChatOpenAI(azure_deployment=deployment, openai_api_version=api_version, temperature=0.0)
    except TypeError:
        return AzureChatOpenAI(deployment_name=deployment, api_version=api_version, temperature=0.0)

@lru_cache(maxsize=1)
def get_embedder():
    """Azure embeddings model used for FAISS vectorization."""
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    api_version = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
    try:
        return AzureOpenAIEmbeddings(azure_deployment=deployment, openai_api_version=api_version)
    except TypeError:
        return AzureOpenAIEmbeddings(deployment=deployment, api_version=api_version)

@lru_cache(maxsize=1)
def get_parser():
    """Shared string output parser for chains."""
    return StrOutputParser()

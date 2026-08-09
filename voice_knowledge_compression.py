from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor 
from langchain_openai import OpenAI

from ap_keys import my_api_key
OPENAI_API_KEY, CORE_API_KEY, ELEVENLABS_API_KEY, GEMINI_API_KEY, COHERE_API_KEY = my_api_key()

def pretty_print_docs(docs):
    return f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)])

def extract_relevant_doc(vector_db, question):
    # Wrap our vectorstore
    llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-5.4", temperature=0.1)
    compressor = LLMChainExtractor.from_llm(llm) 

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_db.as_retriever(search_type = "mmr")  #base_retriever=vector_db.as_retriever()
    )

    compressed_docs = compression_retriever.invoke(question)        
    prettified_document = pretty_print_docs(compressed_docs)
    return prettified_document

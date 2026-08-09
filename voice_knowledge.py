import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
#from elevenlabs.client import ElevenLabs


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)

import pprint
import tempfile
from ap_keys import my_api_key
import os
from typing import List
from openai import OpenAI
from langchain_core.documents import Document
from voice_knowledge_query_expander import *
from voice_knowledge_compression import extract_relevant_doc
from langchain_classic.prompts import PromptTemplate

OPENAI_API_KEY, CORE_API_KEY, ELEVENLABS_API_KEY, GEMINI_API_KEY, COHERE_API_KEY = my_api_key()

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from different file types"""
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(
                directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
            ),
            ".csv": DirectoryLoader(directory, glob="**/*.csv", loader_cls=CSVLoader),
            
        }

        documents = []
        for file_type, loader in loaders.items():
            try:
                documents.extend(loader.load())
                print(f"Loaded {file_type} documents")
            except Exception as e:
                print(f"Error loading {file_type} documents: {str(e)}")

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)

    def create_vector_store(
        self, documents: List[Document], persist_directory: str
    ) -> Chroma:
        """Create and persist vector store if it doesn't exist, otherwise load existing one"""
        # Check if persist_directory exists and has content
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print(f"Loading existing vector store from {persist_directory}")
            # Load existing vector store
            vector_store = Chroma(
                persist_directory=persist_directory, embedding_function=self.embeddings
            )
        else:
            print(f"Creating new vector store in {persist_directory}")
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)

            # Create new vector store
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory,
            )
            vector_store.persist()

        return vector_store
    
class VoiceGenerator:
    def __init__(self):
        #Client is a locally installed Kokoro TTS container on DOcker 
        #For a paid TTS client, consider using ElevenLabs @ https://elevenlabs.io/
        
        self.client = OpenAI(
                    base_url="http://localhost:8880/v1", 
                    api_key="kokoro"
                )
        # Default available voices
        kokoro_voices = [
            "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "af_alloy", "af_aoede", "af_bella", 
            "am_adam", "af_bella+af_sky", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
        ]
        self.available_voices = sorted(kokoro_voices)
        self.default_voice = "af_heart"

    def generate_voice_response(self, text: str, voice_name: str = None) -> str:
        """Generate voice response"""
        try:
            selected_voice = voice_name or self.default_voice

            # Generate audio using the client
            audio_generator = self.client.audio.speech.create(
                            model="kokoro",
                            voice=selected_voice, 
                            input=text,
                            response_format="mp3"
                        )

            # Convert generator to bytes
            audio_bytes = audio_generator.content   # or use .read() for one time access

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                return temp_audio.name

        except Exception as e:
            print(f"Error generating voice response: {e}")
            return None
        
class VoiceAssistantRAG:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-5.4", temperature=0)
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
        self.vector_store = None
        self.qa_chain = None
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator()

    def setup_vector_store(self, vector_store):
        """Initialize the vector store and QA chain"""
        self.vector_store = vector_store

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k":5}, search_type="similarity"),
            memory=memory,
            verbose=True,
        )

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        recording = sd.rec(
            int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1
        )
        sd.wait()
        return recording

    def transcribe_audio(self, audio_array):
        """Transcribe audio using Whisper"""
        import time
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, audio_array, self.sample_rate)
            result = self.whisper_model.transcribe(temp_audio.name)
            time.sleep(2)
            temp_audio.close()
            os.unlink(temp_audio.name)
        return result["text"]

    def generate_response(self, query):
        """Generate response using RAG system"""
        if self.qa_chain is None:
            return "Error: Vector store not initialized"

        response = self.qa_chain.invoke({"question": query})
        return response["answer"]

    def text_to_speech(self, text: str, voice_name: str = None) -> str:
        """Convert text to speech"""
        return self.voice_generator.generate_voice_response(text, voice_name)


def text_summary(input_text):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer

    # 1. Parse the text
    parser = PlaintextParser.from_string(input_text, Tokenizer("english"))

    # 2. Initialize the summarizer
    summarizer_lsa = LsaSummarizer()

    # 3. Generate the summary (2 sentences)
    summary = summarizer_lsa(parser.document, 2)

    # 4. Convert to string
    final_summary = ' '.join([str(sentence) for sentence in summary])   
    
    return final_summary
    
    
def setup_knowledge_base():
    st.title("Knowledge Base Setup")

    doc_processor = DocumentProcessor()

    uploaded_files = st.file_uploader(
        "Upload your documents", accept_multiple_files=True, type=["pdf", "txt", "md", "csv"]
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            temp_dir = tempfile.mkdtemp()

            # Save uploaded files
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            try:
                # Process documents
                documents = doc_processor.load_documents(temp_dir)
                processed_docs = doc_processor.process_documents(documents)

                # Create vector store
                vector_store = doc_processor.create_vector_store(
                    processed_docs, "knowledge_base"
                )

                # Store in session state
                st.session_state.vector_store = vector_store

                st.success(f"Processed {len(processed_docs)} document chunks!")

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
            finally:
                # Cleanup
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
                
def main():
    st.set_page_config(page_title="Voice RAG Assistant", layout="wide")

    # Check for API keys
    #elevenlabs_api_key = ELEVENLABS_API_KEY
    openai_api_key = OPENAI_API_KEY

    if not openai_api_key:
        st.error(
            "Please set OPENAI_API_KEY in your environment variables"
        )
        return

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Setup Knowledge Base", "Voice Assistant"])

    if page == "Setup Knowledge Base":
        vector_store = setup_knowledge_base()
        if vector_store:
            st.session_state.vector_store = vector_store

    else:  # Voice Assistant page
        if "vector_store" not in st.session_state:
            st.error("Please setup knowledge base first!")
            return

        st.title("Voice Assistant RAG System")

        # Initialize assistant
        assistant = VoiceAssistantRAG()
        # Initialize the vector store and QA chain
        assistant.setup_vector_store(st.session_state.vector_store)

        # Voice selection
        try:
            available_voices = assistant.voice_generator.available_voices
            print(f"Available Voices === {available_voices}")
            if available_voices:
                selected_voice = st.sidebar.selectbox(
                    "Select Voice",
                    available_voices,
                    index=(
                        available_voices.index("af_heart")
                        if "af_heart" in available_voices
                        else 0
                    ),
                )
            else:
                st.warning("No voices available. Using default voice.")
                selected_voice = "af_heart"
        except Exception as e:
            st.error(f"Error loading voices: {e}")
            selected_voice = "af_heart"
            st.write(f"Using default voice: {selected_voice}")

        # Recording duration
        duration = st.sidebar.slider("Recording Duration (seconds)", 1, 20, 5)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start Recording"):
                with st.spinner(f"Recording for {duration} seconds..."):
                    audio_data = assistant.record_audio(duration)
                    st.session_state.audio_data = audio_data
                    st.success("Recording completed!")

        with col2:
            if st.button("Process Recording"):
                if "audio_data" not in st.session_state:
                    st.error("Please record audio first!")
                    return

                # Process recording
                with st.spinner("Transcribing..."):
                    query = assistant.transcribe_audio(st.session_state.audio_data)
                    st.write("You said:", query)

                with st.spinner("Generating response..."):
                    try:
                        
                        
                        all_relevant_docs = []
                        query_expander = QueryExpander()           #Import from voice_knowledge_query_expander.py
                        expanded_queries = query_expander.expand_query(query)
                        print(f"EXPANDED QUERY +++++++======: {expanded_queries}")
                        for q in expanded_queries: 
                            compressed_result = extract_relevant_doc(vector_db=st.session_state.vector_store,
                                                                     question=q)
                            all_relevant_docs.append({"q": q, "doc": compressed_result})
                        #pprint.pprint(all_relevant_docs)
                        all_query, all_extracted_doc = [], []
                        for rel_doc in all_relevant_docs:
                            final_summary = text_summary(rel_doc["doc"])
                            all_query.append(rel_doc["q"]); all_extracted_doc.append(final_summary)
                            
                        t = """ You are given four similar {questions} but semantically diverse, 
                            and four {contexts}, each context for a question.
                            Being a very smart and knowledgeable assistant, consider all the questions 
                            and contexts wholistically as a single query. 
                            Based on the detailed wholistic query you have formulated from all questions and contexts, 
                            generate a clear, concise final answer to the original question. Focus on the most important 
                            points while maintaining accuracy.

                            Similar Questions: {questions}

                            Similar diverse semantics: {contexts}

                            Please provide a final answer that:
                            1. Acknowledge the user’s query and express gratitude for the opportunity to assist.
                            2. Directly addresses the wholistic query
                            3. Summarizes the key points
                            4. Is clear and concise
                            5. Maintains the crucial citations 
                            6. Is accurate and devoid of ambiguities
                            7. Use positive language and maintain a supportive tone throughout.
                            8. If applicable, include relevant information or resources that could help further.
                            9. Conclude by inviting any follow-up questions or providing encouragement for the user’s pursuit of information.

                            Final Answer:"""
                        prompt = PromptTemplate(
                            template=t, 
                            input_variables=list(("questions", "contexts"))
                        )
                                                    
                        
                        response = assistant.generate_response(
                            prompt.format(questions=" ".join(all_query),
                                          contexts=" ".join(all_extracted_doc))
                        )
                        
                        # response = assistant.generate_response(query)
                        st.write("Response:", response)
                        st.session_state.last_response = response
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        return

                with st.spinner("Converting to speech..."):
                    audio_file = assistant.voice_generator.generate_voice_response(
                        response, selected_voice
                    )
                    if audio_file:
                        st.audio(audio_file)
                        os.unlink(audio_file)
                    else:
                        st.error("Failed to generate voice response")

        # Display chat history
        if "chat_history" in st.session_state:
            st.subheader("Chat History")
            for q, a in st.session_state.chat_history:
                st.write("Q:", q)
                st.write("A:", a)
                st.write("---")


if __name__ == "__main__":
    main()
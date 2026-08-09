# Parrot
An open-sources voice assistant RAG System, with pretrained LLM and vector knowledge base integration, Maximal Marginal Relevance (MMR) query re-ranking feature, multi-modal workflow, multi-modal document support, and conversational memory integration. No cloud required. On-device voice AI + RAG Knowledge base.

<img width="756" height="546" alt="VoiceRAG_Flowchart" src="https://github.com/user-attachments/assets/0328adf9-9cdf-46be-8022-9749694c2ca0" />

## Dependencies 
The entire scripts are written in Python 3 and on VS Code IDE. The dependencies are all included in requirements.txt and they can be installed using pip with pip install -r requirements.txt . 

## Text-to-Speech Backend Engine
The Text-to-speech architecture which enables the audio response of the **Parrot model** runs entirely on the Kokoro TTS, a text-to-speech model with 82M parameters supporting GPU acceleration. 
For this project, we hosted the Kokoro TTS on a Docker Container. To test and run this project locally, install Docker on your machine and install a pre-built Kokoro-FastAPI image. 

For CPU machines, pull and run this community-maintained Koko-FastAPI Image:
```python
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest      
```
For machines with GPU acceleration, pull and run this CUDA-enables Image: 
```python
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest   
```
The API endpoint at **http://localhost:8880/v1/** is already embedded in the code. Just start the Kokoro TTS container in Docker and run the project. 

## Web Interface
The Parrot model uses **streamlit app** @ <a href="https://streamlit.io" target="_blank">Streamlit</a> streamlit.io to display a GUI. Again,

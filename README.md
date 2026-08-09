# Parrot
An open-sources voice assistant RAG System, with pretrained LLM and vector knowledge base integration, Maximal Marginal Relevance (MMR) query re-ranking feature, multi-modal workflow, multi-modal document support, and conversational memory integration. No cloud required. On-device voice AI + RAG Knowledge base.

<img width="756" height="546" alt="VoiceRAG_Flowchart" src="https://github.com/user-attachments/assets/0328adf9-9cdf-46be-8022-9749694c2ca0" />

## Dependencies 
The entire scripts are written in Python 3 and on VS Code IDE. The dependencies are all included in requirements.txt and they can be installed using pip with pip install -r requirements.txt . 

## Text-to-Speech Backend Engine
The Text-to-speech architecture which enables the audio response of the **Parrot model** runs entirely on the Kokoro TTS, a text-to-speech model with 82M parameters supporting GPU acceleration. 
For this project, we hosted the Kokoro TTS on a Docker Container. To test and run this project locally, install Docker on your machine and install a pre-built Kokoro-FastAPI image. 

```python
def hello():
    print("Hello, world!")   

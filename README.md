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
The API endpoint at **http://localhost:8880/v1/** is already embedded in the scripts. Just start the Kokoro TTS container in Docker and run the project. 

## Web Interface
The Parrot model uses **streamlit app** @ <a href="https://streamlit.io" target="_blank">Streamlit</a> to display a GUI. Again, the required syntax to enable a streamlit web interface is written in the scripts. Upon starting the docker and running the model, the web interface opens up to allow for the upload of files.

## Additional Information 
- Since Parrot uses the OpenAI embedding, you need to have the OpenAI API key to run the model.
- You may delete the uploaded vector knowledge base with pre-trained embeddings, if you wish to start a new knowledge base. Upon running the model, it detects the absence of a vector DB and automatically creates one after uploading your document(s).
- If you prefer a cloud TTS service rather than a locally installed TTS image in a docker container, you may consider ElevenLabs AI (which requires a paid API key). To do this, do the following:
  ```python
  Create an Elevenlabs account at https://elevenlabs.io/ and pay a subscription fee as low as $6 to use the API for TTS services
  Execute this command in your terminal or command prompt: pip install elevenlabs
  In voice_knowledge.py script, add the line: from elevenlabs.client import ElevenLabs
  In the VoiceGenerator class initializer, use this client instead: self.client = ElevenLabs(api_key) where api_key is your Elevenlabs api key. Check the Elevenlabs voice libraries for list of available voices and the voice_id.
  ```
- This is the first version of the model. Subsequently, there will be updates to the model.
  
- 

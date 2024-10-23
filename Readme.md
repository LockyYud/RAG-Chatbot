# RAG Chatbot system

## Front-End

This directory contains the front-end code for the ChatBot UI. It is built using Next.js and includes configurations for ESLint, Tailwind CSS, and PostCSS.

It can be deployed on Vercel or other platforms.

For example, I deployed it on Vercel. Here is the link to the demo: https://rag-chatbot-vie.vercel.app/

## Back-end:

Leverage Colab or other platforms that provide free GPUs to download and use local LLMs. It is built as an .ipynb file (you can write it into a .py file to run locally) using Langchain and models from Hugging Face.

It is deployed publicly via Ngrok. To run it you have to have account Ngrok and run end to end the notebook.

Here are some notebooks included in this project:
|Notebook name|Description|
|-------|-----------|
|`NaiveRAG`|This notebook contains the implementation of Standard RAG. The first version uses the API Python SDK such as OpenAI, Groq, and so on  with Langchain. While the second version utilizes Hugging Face.|
|`SpeculativeRAG`|This notebook contains the implementation of Speculative RAG proposed in Google's reseach paper https://arxiv.org/pdf/2407.08223 using OpenAI Python SDK.

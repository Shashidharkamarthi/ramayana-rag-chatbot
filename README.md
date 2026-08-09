# 🪔 Ramayana Knowledge Assistant

A persona-aware **RAG (Retrieval-Augmented Generation) chatbot** that allows users to explore the Ramayana through the perspectives of **Lord Rama, Lakshmana, and Hanuman**.

The application retrieves relevant passages from a Ramayana knowledge base and uses an AI model to generate answers grounded in those retrieved passages.

## ✨ Features

* 🏹 **Three Ramayana Personas**

  * Lord Rama
  * Lakshmana
  * Hanuman

* 💬 **Separate Chat History**

  * Each persona maintains an independent conversation.
  * Switching between personas restores the previous conversation for that persona.

* 🔎 **RAG-based Question Answering**

  * Retrieves relevant passages from the Ramayana knowledge base.
  * Answers are grounded in the retrieved documents.

* 🤖 **Multiple AI Providers**

  * Google Gemini
  * OpenAI

* 🧠 **Multiple AI Models**

  * Gemini models configured in the application
  * OpenAI models configured in the application

* 🔐 **User API Key**

  * The application allows the user to enter their own API key.
  * API keys are not stored in the source code.

* 🎙️ **Voice Input**

  * Voice questions are supported when Gemini is selected.

* 📚 **Source Passages**

  * The application displays the source documents used to generate an answer.

* 🌿 **Kid-friendly Fallback**

  * When the knowledge base does not contain enough information to answer a question, the chatbot provides suggested questions.

* 🎨 **Attractive Light Theme**

  * Warm cream, saffron, and gold styling.
  * Persona cards and chat bubbles provide a friendly interface.

## 🏗️ Project Structure

```text
ramayana-chatbot/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Ramayana source documents
│
├── prompts/
│   ├── rama.txt
│   ├── lakshmana.txt
│   └── hanuman.txt
│
├── rag_utils/
│   ├── document_loader.py
│   ├── vector_store.py
│   └── rag_pipeline.py
│
└── vector_db/
    ├── index.faiss
    └── index.pkl
```

## 🔄 How It Works

The application follows a Retrieval-Augmented Generation pipeline:

```text
User Question
      │
      ▼
Select Persona
      │
      ▼
Retrieve Relevant Ramayana Passages
      │
      ▼
FAISS Vector Store
      │
      ▼
Persona-specific Prompt
      │
      ▼
Gemini / OpenAI
      │
      ▼
Generated Answer
      │
      ▼
Answer + Sources
```

### 1. Document Loading

Ramayana source documents such as PDF, DOCX, and TXT files are loaded from the `data/` directory.

### 2. Document Chunking

The documents are divided into smaller overlapping chunks so that relevant information can be retrieved efficiently.

### 3. Embeddings

The document chunks are converted into numerical embeddings using the selected AI provider.

### 4. FAISS Vector Store

The embeddings are stored in a FAISS vector database.

The persisted vector store contains:

```text
index.faiss
index.pkl
```

This allows the application to reuse an existing vector store instead of embedding all documents every time.

### 5. Retrieval

When the user asks a question, the application retrieves the most relevant passages from the FAISS vector store.

### 6. Persona-aware Generation

The retrieved passages are combined with a persona-specific system prompt.

The selected persona determines the style and perspective of the response.

### 7. Response

The AI generates an answer based on the retrieved context and displays the answer along with the source passages used.

## 🛠️ Technologies Used

* Python
* Streamlit
* LangChain
* FAISS
* Google Gemini
* OpenAI
* Python-dotenv / environment variables
* PDF, DOCX, and TXT document loaders

## 📦 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd ramayana-chatbot
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 🔐 API Key Setup

The application requires an API key for the selected AI provider.

You can either enter the API key through the application's sidebar or configure it through environment variables as supported by the project.

For security, **never commit API keys to GitHub**.

Do not upload:

```text
.env
```

or files containing:

```text
OPENAI_API_KEY
GOOGLE_API_KEY
```

## ▶️ Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

Choose:

1. AI provider
2. AI model
3. API key
4. Ramayana persona
5. Ask your question

## 💬 Example Questions

### Lord Rama

* Why did you agree to go to the forest for 14 years?
* How did you meet Sita?
* What happened when you fought Ravana?

### Lakshmana

* Why did you go with Rama into the forest?
* What is the Lakshmana Rekha?
* How did you help find Sita?

### Hanuman

* How did you jump across the ocean to Lanka?
* How did you find Sita in Lanka?
* Why did you carry the whole mountain?

## 📚 Knowledge Base

The chatbot uses Ramayana source documents placed inside the `data/` directory.

Additional supported documents can include:

* PDF
* DOCX
* TXT

The documents are automatically loaded and divided into chunks when a new vector store needs to be created.

## ⚠️ Important Notes

* A valid API key is required to generate responses.
* Embedding APIs may have usage limits or quotas.
* Gemini and OpenAI embeddings use different vector spaces, so their vector stores should not be mixed.
* Existing FAISS indexes are reused when available.
* If the source documents are changed significantly, the vector store may need to be rebuilt.
* AI-generated answers may occasionally contain mistakes or incomplete information.

## 🔒 Security

Never commit secret credentials to the repository.

The `.gitignore` file should include:

```text
.env
__pycache__/
*.pyc
```

If an API key is accidentally uploaded to GitHub, revoke it immediately and generate a new key.

## 🎯 Project Objective

The goal of this project is to create an interactive and educational way to learn about the Ramayana using modern AI techniques.

By combining **RAG, vector search, persona-based prompting, and conversational AI**, users can explore Ramayana stories and teachings through different character perspectives.

## 👨‍💻 Project

**Ramayana Knowledge Assistant**

Built using Python, Streamlit, LangChain, FAISS, Gemini, and OpenAI.

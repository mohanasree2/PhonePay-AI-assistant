#  PhonePay RAG AI Assistant

An advanced, end-to-end Retrieval-Augmented Generation (RAG) system built to track billing information, manage transactional alerts, and answer complex financial platform queries. The system leverages **Google Gemini LLM**, **LangChain orchestration**, and a local **Chroma vector database** to provide intelligent contextual answers about payments, UPI protocols, investments, and custom user invoice states.

---

##  System Architecture & Workflow

The framework operates through an isolated data ingestion, vector indexing, and query execution pipeline:
```text
[ data.json ] ───┐
[ faq_data ] ────┼─► [ src/ingest.py ] ─► [ Embeddings ] ─► [ chroma_db/ ]
[ text_kb  ] ────┘                                                │
▼
[ User Query ] ──► [ app.py ] ──► [ src/rag_chain.py ] ◄─── [ Context ]
│                 │
▼                 ▼
[ UI Dashboards ] ◄── [ Gemini LLM ]
```
1. **Knowledge Ingestion (`src/ingest.py`):** Parses structured JSON invoice data (`data.json`), platform text bases, and FAQs. It segments them into semantic chunks, generates vector embeddings using `sentence-transformers`, and stores them locally in a persistent **ChromaDB** database.
2. **Context Retrieval (`src/rag_chain.py`):** When a user types a query into the UI, the system runs a similarity search over the vector space to extract the most relevant documents.
3. **Response Assembly:** The retrieved text chunks are injected into a prompt layout along with the user's question, which **Google Gemini** converts into a human-like, accurate reply.
4. **Alerts & Reminders (`src/reminder_engine.py`):** Dynamically parses upcoming invoice due dates from your transaction data log files to display alert indicators across the web dashboard.

---

## 📂 Project Structure

```text
phonepe-rag/
├── .streamlit/
│   └── config.toml             # Global UI appearance configuration
├── data/
│   ├── data.json               # Custom structured user bills and ledger file
│   ├── faq_data.txt            # Frequently asked customer questions
│   └── phonepay_knowledge_base.txt # Core platform platform rules/documentation
├── src/
│   ├── ingest.py               # Document parsing & Vector Database builder
│   ├── rag_chain.py            # LangChain orchestrator & Gemini integration
│   └── reminder_engine.py      # Background date parser and alert engine
├── app.py                      # Main Streamlit UI frontend controller
├── requirements.txt            # Pinpointed project dependencies
├── setup_and_run.py            # Master health checker and orchestration script
└── .env                        # Local environment variables and secrets
```
## Installation & Setup Instructions
Python 3.11 is required to match package compatibility and ensure Streamlit builds flawlessly.
An active Google AI Studio Gemini API Key.

1. Isolate Environment via Virtual Sandbox (venv)
```bash
# 1. Open your terminal in your workspace folder
cd "C:\Users\User\Downloads\phonepe rag"

# 2. Build the virtual environment folder
python -m venv venv

# 3. Activate the sandbox environment
# On Windows PowerShell:
venv\Scripts\activate
```
2. Install Pinpointed Requirements
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
3. Setup Secrets Configuration File
Create a .env file in your root folder and add your specific environment keys:
```Code snippet
GOOGLE_API_KEY="your_actual_gemini_api_key_here"
CHROMA_DB_PATH="./chroma_db"
KNOWLEDGE_BASE_PATH="./data/phonepay_knowledge_base.txt"
FAQ_DATA_PATH="./data/faq_data.txt"
USER_BILLS_PATH="./data/data.json"
```
## Execution & Hotfixes Implemented
This project includes custom architecture patches applied directly to make runtime integration seamless:

1. Critical Module Paths Fix
To allow the frontend dashboard (app.py) to smoothly read custom back-end modules from the nested src/ folder, the workspace environment root is dynamically expanded on execution initialization inside app.py:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
```
2. Theme & Text Contrast Correction
To resolve text color clipping caused by Streamlit operating in a dark system browser setting, background message wrappers inside app.py are explicitly forced to high-contrast readable dark gray colors (#202124):
```CSS
.user-message, .assistant-message, .bill-card-urgent {
    background: #f0e6ff;
    color: #202124; /* Prevents white text on white box bugs */
}
```
3. Run the Automation Script
Launch the master configuration checker tool from your active terminal session:

```Bash
python setup_and_run.py
```
When prompted by the system:
```Plaintext
Rebuild vector store? (y/n, default=n): y or yes
```
Type y and hit Enter to wipe old storage structures, index your custom data.json records freshly,
## Production Deployment Guidelines (GitHub Exclusions)
When migrating this repository to hosting spaces like Streamlit Community Cloud, never expose your active keys or build cache cache models. Ensure the following properties are added to a .gitignore file:

```Plaintext
chroma_db/    # Built automatically on the host server via setup scripts
venv/         # Isolated virtual environments are ignored
logs/         # Local diagnostics ignored
.env          # CRITICAL SECURITY RISK - Never push secrets to GitHub
```
Note: During cloud deployment configuration setup, enter all environmental strings listed in your .env file into the host platform's secure Advanced Settings/Secrets console instead.

### here you see the deployment of this app
Loacl streamlit deployment: http://localhost:8501/
Hugging Face deployment: https://huggingface.co/spaces/Mohanasree-2/Phonepay_AI_Assistant-v2/tree/main

Here i created duplicate space and saved my key in private

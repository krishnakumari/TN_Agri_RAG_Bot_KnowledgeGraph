# 🌾 Tamil Nadu Agriculture Schemes GraphRAG Assistant

An end-to-end **GraphRAG (Knowledge Graph + Retrieval-Augmented Generation)** assistant built to help farmers and agricultural officers query official schemes from the **Tamil Nadu Agriculture & Farmers Welfare Department** (`tn.gov.in`).

Unlike standard chunk-based vector RAG, this system extracts structured entities (`Scheme`, `Crop`, `District`, `Benefit`, `Authority`, `BeneficiaryGroup`) into a **NetworkX Knowledge Graph**, enforces strict topic & jailbreak safety via **NVIDIA NeMo Guardrails**, serves responses through **FastAPI**, renders an interactive **Streamlit + PyVis** web interface, and validates accuracy using **DeepEval**.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[TN Govt Portal<br/>tn.gov.in] -->|scraper.py<br/>BeautifulSoup4| B[(tn_agri_schemes.json<br/>54 Official Schemes)]
    B -->|build_kg.py<br/>LangChain + OpenAI| C[(tn_agri_kg.json<br/>113 Nodes • 346 Edges)]
    
    U[🧑‍🌾 Farmer / User] -->|Streamlit UI<br/>app.py| API[FastAPI Backend<br/>api.py]
    API --> NEMO{NVIDIA NeMo Guardrails<br/>Input & Output Rails}
    
    NEMO -->|Off-Topic / Injection| BLOCK[Polite Refusal Response]
    NEMO -->|Valid Agri Query| RAG[GraphRAG Engine<br/>graph_rag.py]
    
    RAG <-->|Entity Extraction &<br/>1-Hop Traversal| C
    RAG -->|Grounded Synthesis| LLM[OpenAI gpt-4o-mini]
    LLM --> NEMO
    NEMO --> API
    
    EVAL[DeepEval Suite<br/>evaluate_rag.py] -.->|Faithfulness &<br/>Answer Relevancy| RAG
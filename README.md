# Scholar Inbox: Personalized Paper Recommendations for Scientists

This project is a high-fidelity replication (**~96%**) of the research paper **"Scholar Inbox: Personalized Paper Recommendations for Scientists"** (arXiv:2504.08385v2).

It provides a local-first paper recommendation platform that collects data from arXiv, visualizes papers in a 2D embedding space with t-SNE, and learns user preferences with a weighted logistic regression model.

---

## Key Features

1. **Daily Digest**
   - Ranks papers by relevance on a score scale of $[-100, 100]$.
   - Highlights the most important contribution or idea sentences in each abstract for speed reading.
   - **View PDF & AI Chat Assistant (Neural RAG)**: opens the official arXiv PDF inside the app and provides a RAG chat assistant with hierarchical **Parent-Child Chunking**. Semantic retrieval uses **Google `text-embedding-004` (768-dimensional)** with **batching** and **disk caching**, with local TF-IDF fallback.
   - Suggests similar papers using real-time cosine similarity.
   - Supports BibTeX preview and one-click reference copy.

2. **Scholar Maps**
   - Visualizes the paper collection in 2D with t-SNE.
   - High-performance Canvas map with drag-to-pan and scroll-to-zoom.
   - Shows dynamic hierarchical topic labels based on zoom level (broad field → subfield → paper title).
   - **Diversity Active Learning (K-Means)**: runs K-Means clustering ($k=5$) on 20 uncertainty-boundary papers (probability $\approx 0.5$) to recommend 5 papers from 5 different research groups, helping the model learn broader user preferences faster.

3. **Collections & Search**
   - Searches papers using keywords or free-form text (semantic and lexical search).
   - Lets users create personal paper folders.
   - Suggests collection expansion based on the average vector of items inside a folder.

4. **Conference Planner**
   - Ranks poster sessions by average relevance to user preferences.
   - Adds posters to a detailed personal schedule with quick links for **PDF reading and AI RAG Q&A**.

5. **AI Research Workspace**
   - **LangGraph Multi-Agent Workflows**: sets up an automated agent pipeline (ArxivSearchAgent → parallel PaperCriticAgent workers → LiteratureReviewAgent).
   - **Phased Technical Roadmap**: generates a 3-stage Vietnamese research plan (baseline replication → module integration → evaluation), milestones, and hardware/software risk tracking.
   - **Custom Markdown-to-JSX Compiler**: renders interactive markdown in the UI and builds dynamic HTML/CSS glassmorphism tables.
   - **Hybrid API Router (Gemini & Groq)**: automatically routes API keys to fallback services for high availability.

---

## Technology Stack

- **Backend**: Python 3.12, Flask, Flask-CORS, LangGraph, NumPy, scikit-learn (TfidfVectorizer, LogisticRegression, t-SNE), Google Gemini API (`embed-004`), Groq Llama 3.3.
- **Frontend**: React 19 (Vite), Lucide React, HTML5 Canvas.
- **Containerization**: Docker, Docker Compose.

---

## How to Run

### Option 1: Docker (Recommended)
Run the following command from the project root:
```bash
docker compose up --build
```

This starts both services automatically:
- **Frontend SPA**: [http://localhost:5173/](http://localhost:5173/)
- **Backend API**: [http://localhost:5001/](http://localhost:5001/)

---

### Option 2: Run Locally

#### 1. Start the Backend API (Flask)
Run the backend with Python 3.12:
```bash
python backend/app.py
```

On the first run, the system will fetch the latest 200 papers from the arXiv API and compute t-SNE coordinates.

#### 2. Start the Frontend (React + Vite)
Open a new terminal in the project directory and run:
```bash
npm run dev
```

#### 3. Open the App
Visit the app in your browser at [http://localhost:5173/](http://localhost:5173/)

---

## Technical Documentation

For architecture details, code structure, data flow, and the weighted learning math used in the paper, see:
[TECHNICAL.md](./TECHNICAL.md)

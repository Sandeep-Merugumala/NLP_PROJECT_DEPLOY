import os
import json
import urllib.request
import urllib.error
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Define the Agent State
class AgentState(TypedDict):
    query: str
    api_key: str
    papers: List[Dict[str, Any]]
    agent_logs: List[Dict[str, Any]]
    retrieved_contexts: List[Dict[str, Any]]
    final_report: str

# 2. General LLM Calling Helper
def call_gemini(prompt: str, api_key: str, model: str = "gemini-flash-lite-latest") -> str:
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return ""
        
    if api_key.startswith("gsk_"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode('utf-8')
            except Exception:
                pass
            print(f"Error calling Groq in agent (HTTPError {e.code}): {e.reason} - Body: {err_body}", flush=True)
            return f"Error: Failed to fetch response from Groq. Details: HTTP Error {e.code}: {e.reason}. Response: {err_body}"
        except Exception as e:
            print(f"Error calling Groq in agent: {e}", flush=True)
            return f"Error: Failed to fetch response from Groq. Details: {e}"
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        headers = {"Content-Type": "application/json"}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data['candidates'][0]['content']['parts'][0]['text']
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode('utf-8')
            except Exception:
                pass
            print(f"Error calling Gemini in agent (HTTPError {e.code}): {e.reason} - Body: {err_body}", flush=True)
            return f"Error: Failed to fetch response from Gemini. Details: HTTP Error {e.code}: {e.reason}. Response: {err_body}"
        except Exception as e:
            print(f"Error calling Gemini in agent: {e}", flush=True)
            return f"Error: Failed to fetch response from Gemini. Details: {e}"

# 3. Agent 1: ArxivSearchAgent (Hybrid Semantic-Personalized Search)
def search_arxiv_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    logs = list(state.get("agent_logs", []))
    
    logs.append({
        "agent": "ArxivSearchAgent",
        "message": f"Analyzing user task query: '{query}'. Initiating hybrid semantic-personalized search on candidate index..."
    })
    
    # Paths setup
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    papers_path = os.path.join(backend_dir, "data", "papers.json")
    ratings_path = os.path.join(backend_dir, "data", "ratings.json")
    
    matched_papers = []
    
    # 1. Load papers and ratings
    all_papers = []
    if os.path.exists(papers_path):
        try:
            with open(papers_path, "r") as f:
                all_papers = json.load(f)
        except Exception as e:
            logs.append({"agent": "ArxivSearchAgent", "message": f"Error loading papers database: {e}"})
            
    user_ratings = {}
    if os.path.exists(ratings_path):
        try:
            with open(ratings_path, "r") as f:
                user_ratings = json.load(f)
        except Exception:
            pass
            
    if all_papers:
        # Create papers lookup index
        papers_lookup = {p["id"]: p for p in all_papers}
        
        # 2. Process personalization rating signals (Upvotes per category)
        upvoted_categories = {}
        for paper_id, rating in user_ratings.items():
            if rating == 1 and paper_id in papers_lookup:
                cat = papers_lookup[paper_id]["primary_category"]
                upvoted_categories[cat] = upvoted_categories.get(cat, 0) + 1
                
        # 3. Compute TF-IDF Semantic similarity of the query against all abstracts
        corpus = [p["title"] + " " + p["abstract"] for p in all_papers]
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            vectorizer.fit(corpus)
            query_vec = vectorizer.transform([query]).toarray().flatten()
            
            scored_candidates = []
            for paper in all_papers:
                paper_vec = vectorizer.transform([paper["title"] + " " + paper["abstract"]]).toarray().flatten()
                similarity = float(np.dot(paper_vec, query_vec))
                
                # Apply category boost (1.0 + 0.15 per upvote in this category)
                cat = paper["primary_category"]
                boost = 1.0 + 0.15 * upvoted_categories.get(cat, 0)
                final_score = similarity * boost
                
                scored_candidates.append({
                    "paper": paper,
                    "similarity": similarity,
                    "boost": boost,
                    "score": final_score
                })
                
            # Rank candidates by final score
            ranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
            matched_papers = [x["paper"] for x in ranked_candidates[:3]]
            
            # Log specific details of selection
            for x in ranked_candidates[:3]:
                logs.append({
                    "agent": "ArxivSearchAgent",
                    "message": f"ArxivSearchAgent: Matched candidate '{x['paper']['title'][:40]}...' (Similarity: {x['similarity']:.2f}, Personalization Boost: {x['boost']:.2f})."
                })
                
        except Exception as e:
            logs.append({"agent": "ArxivSearchAgent", "message": f"Semantic query vectorization failed: {e}. Falling back to default list."})
            matched_papers = all_papers[:3]
            
    if not matched_papers:
        matched_papers = all_papers[:2]
        
    logs.append({
        "agent": "ArxivSearchAgent",
        "message": f"ArxivSearchAgent to PaperCriticAgent: I have dispatched the top {len(matched_papers)} relevant publications to your node. Critic Agent, please extract their mathematical cores, empirical settings, and limitations."
    })
    
    return {
        "papers": matched_papers,
        "agent_logs": logs
    }

# 4. Agent 2: PaperCriticAgent (Structured Critical Extraction)
def critic_papers_node(state: AgentState) -> Dict[str, Any]:
    papers = state["papers"]
    api_key = state["api_key"]
    logs = list(state.get("agent_logs", []))
    retrieved = []
    
    if not papers:
        logs.append({
            "agent": "PaperCriticAgent",
            "message": "PaperCriticAgent to ArxivSearchAgent: Alert - No candidate papers received at node. Terminating."
        })
        return {"agent_logs": logs, "retrieved_contexts": []}
        
    logs.append({
        "agent": "PaperCriticAgent",
        "message": "PaperCriticAgent to ArxivSearchAgent: Received candidates. Initializing deep PDF extraction workflows from cache..."
    })
        
    import concurrent.futures

    def process_single_paper(p):
        # Load PDF RAG chunks
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_cache_dir = os.path.join(backend_dir, "pdf_cache")
        cache_file = os.path.join(pdf_cache_dir, f"{p['id']}.json")
        summary_text = ""
        
        chunks = []
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    chunks = json.load(f)
            except Exception:
                pass
                
        if chunks:
            # Join top chunks to build context
            summary_text = " ".join([c["text"] for c in chunks[:4]])
        else:
            summary_text = p["abstract"]
            
        if api_key:
            prompt = f"""
You are an expert academic peer reviewer. Perform an in-depth, highly structured scientific critique of this paper:
Title: {p['title']}
Context:
{summary_text[:3500]}

Please extract and analyze the following aspects thoroughly, using clear markdown lists (-), bold text (**), and headings (##, ###) where appropriate:
1. Mathematical & Algorithmic Core:
   - Identify the exact mathematical foundations, loss functions, algorithms, and key structural modules.
   - Explain how the core process flow works.
2. Empirical Setup & Metrics:
   - Identify the specific datasets, baseline models compared against, training parameters (learning rates, batch sizes, hardware/GPU requirements).
   - State the primary evaluation metrics and key results.
3. Scientific Limitations & Constraints:
   - Detail the primary limitations, compute constraints, failure modes, or narrow assumptions.

Format your critique cleanly in English. Be highly technical, specific, and precise.
"""
            critique = call_gemini(prompt, api_key)
        else:
            # Advanced local NLP mining (extracting numbers and parameters using regex patterns)
            epochs_match = re.findall(r'\b\d+\s*(?:epochs|epoch)\b', summary_text, re.IGNORECASE)
            batch_match = re.findall(r'\b(?:batch\s*size|batch)\s*(?:of\s*)?\d+\b', summary_text, re.IGNORECASE)
            lr_match = re.findall(r'\b(?:learning\s*rate|lr)\s*(?:of\s*)?(?:0\.\d+|\d+e-\d+)\b', summary_text, re.IGNORECASE)
            
            critique = "1. Mathematical & Algorithmic Core\n"
            critique += f"- Core Methodology: {p.get('highlight', 'arXiv algorithms')}.\n"
            critique += "- Uses token embeddings and high-dimensional semantic mapping space.\n"
            
            critique += "\n2. Empirical Setup & Metrics\n"
            critique += f"- Primary Category: {p['category_name']} ({p['primary_category']}).\n"
            if epochs_match:
                critique += f"- Extracted training parameter: {', '.join(epochs_match)}.\n"
            if batch_match:
                critique += f"- Extracted batch parameter: {', '.join(batch_match)}.\n"
            if lr_match:
                critique += f"- Extracted learning rate: {', '.join(lr_match)}.\n"
                
            critique += "\n3. Scientific Limitations & Constraints\n"
            critique += "- Quadratic computation complexity concerning input context lengths.\n"
            critique += "- Scalability limited by vector distance metric calculations."
            
        return {
            "id": p["id"],
            "title": p["title"],
            "critique": critique
        }

    for p in papers:
        logs.append({
            "agent": "PaperCriticAgent",
            "message": f"PaperCriticAgent: Analyzing paper '{p['title'][:45]}...'. Mining mathematical equations, dataset specs, and training metrics."
        })

    # Execute paper analyses concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(papers)) as executor:
        retrieved = list(executor.map(process_single_paper, papers))

    logs.append({
        "agent": "PaperCriticAgent",
        "message": "PaperCriticAgent to LiteratureReviewAgent: Critiques completed. I noted several baseline constraints. Sending structured parameters for final comparative synthesis."
    })
    
    return {
        "agent_logs": logs,
        "retrieved_contexts": retrieved
    }

# 5. Agent 3: LiteratureReviewAgent (Comparative Academic Synthesis)
def synthesize_report_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    retrieved = state["retrieved_contexts"]
    api_key = state["api_key"]
    logs = list(state.get("agent_logs", []))
    
    logs.append({
        "agent": "LiteratureReviewAgent",
        "message": "LiteratureReviewAgent to PaperCriticAgent: Reviews received. Correlating the baseline parameters and drafting the comparative synthesis table..."
    })
    
    if not retrieved:
        report = "### Comparative Synthesis Fail\n\nNo structured critiques found to synthesize. Please search another research topic."
        logs.append({
            "agent": "LiteratureReviewAgent",
            "message": "Failed to compile report: Empty critiques array."
        })
        return {"agent_logs": logs, "final_report": report}
        
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    papers_path = os.path.join(backend_dir, "data", "papers.json")
    papers_lookup = {}
    if os.path.exists(papers_path):
        try:
            with open(papers_path, "r") as f:
                all_papers = json.load(f)
                papers_lookup = {p["id"]: p for p in all_papers}
        except Exception:
            pass
            
    critiques_str = ""
    for r in retrieved:
        critiques_str += f"\nPaper: {r['title']}\nCritique:\n{r['critique']}\n"
        
    if api_key:
        prompt = f"""
You are a Director of Research. Build a detailed Research Roadmap and Implementation Plan based on the research topic: '{query}' and the paper analyses provided below.

Output quality requirements:
- **Academic depth:** Analyze the core algorithms, mathematical mechanisms (e.g., loss functions, probability distributions, Markov chains) of the reference papers. Avoid generic introductions.
- **Practical feasibility:** Provide specific technical steps (library configurations, CUDA versions, GPU hardware requirements) for experimental deployment.
- **Sharp comparative analysis:** Identify contradictions in assumptions or complementary synergies between the methods.

The roadmap report must use standard Markdown format (headings #, ##, ###, bullet lists -, and comparison tables) and must include all of the following sections:

# Research Roadmap & Implementation Plan: {query}

## 1. Overview & Strategic Objectives
- Detailed introduction to the scientific context and current technical challenges of the topic '{query}'.
- Define specific research objectives and success metrics (Key Results) the roadmap targets.

## 2. Reference Mapping (Comparative Matrix)
- Detailed analysis of the role of the 3 reference papers in solving the target problem.
- Draw a Markdown comparison table with columns: Paper | Core Method (clearly describe the model/algorithm) | Experimental Metrics (specify datasets and results achieved) | Weaknesses/Limitations (specify hardware limits, latency, or edge-case failures).

## 3. Phased Technical Implementation Plan
- **Phase 1: Foundation & Replication (Weeks 1-3)**: Specify which papers need repo cloning and replication first, framework versions (e.g., PyTorch, TensorFlow), and sample datasets to run.
- **Phase 2: Synergy & Integration (Weeks 4-6)**: Propose specific technical solutions to combine architectures (e.g., embedding Paper A's encoder into Paper B's diffusion model) to optimize performance or reduce computational complexity.
- **Phase 3: Training & Evaluation (Weeks 7-9)**: Fine-tuning procedures (Fine-tuning, LoRA), baseline setup, and benchmark metrics (e.g., accuracy, F1, latency, FPS) to monitor.

## 4. Milestone Timeline
- Create a Markdown table breaking down the plan by week/month, specifying: Timeline | Detailed Activities | Deliverables.

## 5. Technical Risk Assessment & Mitigation
- Analyze the 3 largest technical risks (e.g., loss non-convergence, GPU memory shortage, overfitting on narrow datasets) based on limitations of the original papers.
- Propose detailed technical contingency plans for each risk (e.g., FP16/BF16, gradient checkpointing, or transfer learning).

Paper analysis information:
{critiques_str}

Write the report in a scientific, professional tone with rigorous logical reasoning, entirely in English.
"""
        report = call_gemini(prompt, api_key)
    else:
        # Local Heuristic Comparative Matrix and Synthesis
        report = f"# Literature Review\n\n"
        report += f"# Research Roadmap & Implementation Plan: {query}\n\n"
        report += "*(Fallback Scientific Roadmap - Gemini/Groq API Key not configured)*\n\n"
        report += "## Introduction\n"
        report += f"This literature review outlines a practical and research-driven roadmap for '{query}'.\n\n"
        report += "## Methodology\n"
        report += "Hybrid local retrieval with critique synthesis and comparative planning based on available paper contexts.\n\n"
        report += "## 1. Overview & Strategic Objectives\n"
        report += f"This roadmap aims to establish a comprehensive experimental pipeline addressing the topic '{query}' by combining cutting-edge research.\n\n"
        report += "## 2. Reference Mapping (Comparative Matrix)\n"
        report += "| Paper | Core Method | Experimental Metrics | Weaknesses/Limitations |\n"
        report += "| --- | --- | --- | --- |\n"
        for r in retrieved:
            p_obj = papers_lookup.get(r["id"], {})
            report += f"| {r['title'][:30]}... | {p_obj.get('primary_category', 'N/A')} | {p_obj.get('highlight', 'N/A')[:40]}... | High computational resource requirements |\n"
        report += "\n"
        report += "## 3. Phased Implementation Plan\n"
        report += "- **Phase 1: Foundation & Replication**: Set up and experiment with the open-source codebases accompanying the research papers to verify correctness.\n"
        report += "- **Phase 2: Integration & Synergy**: Resolve input/output conflicts between models and build integration modules.\n"
        report += "- **Phase 3: Optimization & Evaluation**: Fine-tune hyperparameters and cross-evaluate performance.\n\n"
        report += "## 4. Milestone Schedule\n"
        report += "| Timeline | Key Activities | Expected Outcomes |\n"
        report += "| --- | --- | --- |\n"
        report += "| Weeks 1-2 | Set up virtual environment and re-run paper source code | Standardized working environment |\n"
        report += "| Weeks 3-4 | Execute independent tests for each original module | Baseline metrics for comparison |\n"
        report += "| Weeks 5-8 | Develop and integrate model components | First integrated prototype (MVP) |\n"
        report += "\n"
        report += "## 5. Technical Risks & Contingency Plans\n"
        report += "- **Risk 1**: GPU memory overflow during integrated model training. *Contingency*: Apply parameter-efficient fine-tuning (LoRA, QLoRA) or reduce batch size.\n"
        report += "- **Risk 2**: Real-world data distribution mismatch (Domain shift). *Contingency*: Apply Domain Adaptation techniques or data augmentation training."
        
    logs.append({
        "agent": "LiteratureReviewAgent",
        "message": "LiteratureReviewAgent: Synthesis report successfully drafted. Literature review, comparative table, and research directions are final. Closing workflow graph."
    })
    
    return {
        "agent_logs": logs,
        "final_report": report
    }

# 6. Build the LangGraph Workflow State Machine
def run_agent_workflow(query: str, api_key: str) -> Dict[str, Any]:
    # Initialize StateGraph
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("search_arxiv", search_arxiv_node)
    workflow.add_node("critic_papers", critic_papers_node)
    workflow.add_node("synthesize_report", synthesize_report_node)
    
    # Establish Edges
    workflow.set_entry_point("search_arxiv")
    workflow.add_edge("search_arxiv", "critic_papers")
    workflow.add_edge("critic_papers", "synthesize_report")
    workflow.add_edge("synthesize_report", END)
    
    # Compile Graph
    app = workflow.compile()
    
    # Run Graph execution
    initial_state = {
        "query": query,
        "api_key": api_key,
        "papers": [],
        "agent_logs": [],
        "retrieved_contexts": [],
        "final_report": ""
    }
    
    final_output = app.invoke(initial_state)
    return final_output

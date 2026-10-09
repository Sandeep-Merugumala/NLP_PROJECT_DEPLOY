import os
import json
import urllib.request
import urllib.error
import re

# --- Current Google Generative AI model names ---
# The older gemini-2.0-flash / gemini-2.5-flash / text-embedding-004 models have
# been retired by Google and now return HTTP 404. These constants keep every call
# site pointed at a supported model.
GEMINI_GEN_MODEL = "gemini-flash-lite-latest"
GEMINI_EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768


def _gemini_gen_url(api_key):
    return f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GEN_MODEL}:generateContent?key={api_key}"

def get_keywords(text):
    """
    Extracts meaningful keywords from text by filtering stop words.
    """
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'of', 'in', 'on', 'at', 
        'to', 'for', 'with', 'by', 'about', 'against', 'between', 'into', 'through', 'during', 
        'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
        'can', 'will', 'just', 'don', 'should', 'now', 'using', 'reusing', 'optimal', 'speed', 
        'accuracy', 'single', 'multi', 'composed', 'task'
    }
    # Clean and split
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in stop_words}

def explain_recommendation_heuristic(user_interests, paper_title, paper_abstract, category_name):
    """
    Local NLP heuristic that performs keyword overlapping to explain
    relevance and generate a summary when Gemini API is offline.
    """
    # Combine upvoted paper titles for interest keywords
    interest_text = " ".join(user_interests)
    interest_keywords = get_keywords(interest_text)
    
    # Target paper keywords
    paper_text = f"{paper_title} {paper_abstract}"
    paper_keywords = get_keywords(paper_text)
    
    # Calculate overlap
    overlap = interest_keywords.intersection(paper_keywords)
    matched_words = list(overlap)[:3] # Grab top 3 overlaps
    
    # Generate explanation
    if matched_words:
        keywords_str = ", ".join(f"'{w}'" for w in matched_words)
        explanation = f"Recommended because you liked papers matching keyword(s) {keywords_str}, which are core to this {category_name} research."
        tailored_summary = f"This work advances techniques in {category_name} focusing on {matched_words[0]} mechanisms. It aligns with your interest in {keywords_str} optimizations."
    else:
        explanation = f"Recommended based on your interest in {category_name} models, sharing general concepts with your bookmarked publications."
        tailored_summary = f"This paper presents new methodologies within {category_name}. It proposes optimizations that run parallel to your reading habits."
        
    return {
        "explanation": explanation,
        "tailored_summary": tailored_summary,
        "source": "Local Heuristic"
      }

def explain_recommendation_llm(user_interests, paper_title, paper_abstract, category_name, api_key):
    """
    Calls Google Gemini or Groq Llama API to get structured JSON recommendations explanation.
    """
    prompt = f"""
You are an expert scientific paper recommendation assistant for researchers.
The user has expressed strong interest in these papers they upvoted:
{chr(10).join(f"- {title}" for title in user_interests)}

Now, the user is looking at this recommended paper:
Title: {paper_title}
Category: {category_name}
Abstract: {paper_abstract}

Your tasks:
1. "explanation": Write a detailed, personalized explanation (in English, 2-3 sentences) detailing why this paper matches their upvoted interest profile. Be specific about common concepts, models, or datasets. Do not use any markdown formatting characters (like *, **, #) in the explanation string.
2. "tailored_summary": Write a detailed tailored summary (in English, 3-4 sentences) of this paper tailored to their background, highlighting the most advanced methodologies, dataset scale, or key metrics of this paper that directly align with their research interests. Do not use any markdown formatting characters (like *, **, #) in the summary string.

You must respond in strict JSON format matching the schema. Do not write markdown blocks or HTML.
"""
    
    if api_key.startswith("gsk_"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
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
                text_response = res_data['choices'][0]['message']['content']
                parsed_json = json.loads(text_response.strip())
                parsed_json["source"] = "Groq Llama"
                return parsed_json
        except Exception as e:
            print(f"Error calling Groq API in explain_recommendation_llm: {e}. Falling back to heuristic...")
            return explain_recommendation_heuristic(user_interests, paper_title, paper_abstract, category_name)
    else:
        url = _gemini_gen_url(api_key)
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "explanation": {"type": "STRING"},
                        "tailored_summary": {"type": "STRING"}
                    },
                    "required": ["explanation", "tailored_summary"]
                }
            }
        }
        headers = {"Content-Type": "application/json"}
        try:
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers=headers, 
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                text_response = res_data['candidates'][0]['content']['parts'][0]['text']
                parsed_json = json.loads(text_response.strip())
                parsed_json["source"] = "Gemini LLM"
                return parsed_json
        except Exception as e:
            print(f"Error calling Gemini API: {e}. Falling back to heuristic...")
            return explain_recommendation_heuristic(user_interests, paper_title, paper_abstract, category_name)

def explain_recommendation(user_interests, paper_title, paper_abstract, category_name, api_key=None):
    """
    Main entrypoint for recommendations explanations. 
    Checks for passed api_key or GEMINI_API_KEY environment variable.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        
    if not api_key:
        # Fallback instantly if no API key is provided
        return explain_recommendation_heuristic(user_interests, paper_title, paper_abstract, category_name)
        
    return explain_recommendation_llm(user_interests, paper_title, paper_abstract, category_name, api_key)

def get_gemini_embedding(text, api_key):
    """
    Fetches 768-dimensional dense vector embeddings using Google's text-embedding-004 model.
    Returns None on failure so the application falls back to local TF-IDF similarity.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent?key={api_key}"
    payload = {
        "model": f"models/{GEMINI_EMBED_MODEL}",
        "content": {
            "parts": [{"text": text}]
        },
        "outputDimensionality": EMBED_DIM
    }
    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data['embedding']['values']
    except Exception as e:
        print(f"Error fetching Gemini embedding: {e}. Falling back...")
        return None

def get_gemini_embeddings_batch(texts, api_key):
    """
    Fetches dense vector embeddings in a single batch request using Gemini's text-embedding-004 model.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:batchEmbedContents?key={api_key}"
    requests = []
    for txt in texts:
        requests.append({
            "model": f"models/{GEMINI_EMBED_MODEL}",
            "content": {
                "parts": [{"text": txt}]
            },
            "outputDimensionality": EMBED_DIM
        })
    payload = {"requests": requests}
    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return [emb['values'] for emb in res_data['embeddings']]
    except Exception as e:
        print(f"Error fetching batch Gemini embeddings: {e}")
        return None

def answer_pdf_question_heuristic(chunks, question):
    """
    Local heuristic fallback that displays retrieved snippets from the PDF
    when Gemini API key is offline or unconfigured.
    """
    ans = "🤖 [Local Fallback Mode] [Local Index Fallback]\n\n"
    ans += "Gemini API Key is not configured or has expired/exceeded quota. Below are the most relevant sections found from the PDF document:\n\n"
    for i, c in enumerate(chunks[:2]):
        cleaned_text = c['text'].replace('\n', ' ').strip()
        score = c.get('score', 0.0)
        ans += f"📄 Section {i+1} (Similarity score: {score:.2f}):\n"
        ans += f"\"... {cleaned_text[:350]} ...\"\n\n"
    ans += "💡 Tip: Please enter a valid Gemini API Key in the settings menu (🔑 icon in the top right corner) to unlock full and intelligent AI answers!"
    return ans

def answer_pdf_question(chunks, question, api_key):
    """
    Asks Gemini or Groq to answer a user's question using the retrieved PDF chunks as context.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        
    if not api_key:
        return answer_pdf_question_heuristic(chunks, question)
        
    snippets = ""
    for i, c in enumerate(chunks):
        snippets += f"\nSnippet {i+1} (Page {c['page_idx']}):\n{c['text']}\n"
        
    prompt = f"""
You are a professional scientific research assistant.
Please answer the user's question about the scientific paper based on the provided text citations (Context Snippets) from the paper's PDF file.

Answering Requirements:
1. Visual and readable: You are ALLOWED to use markdown formatting such as bolding (**) or bullet points (-) to highlight key terms, solutions, and methods.
2. Scientific basis: Clearly state which page of the paper the information was extracted from (e.g., [Page X]) based on the "Page X" information in the snippets.
3. Detailed and scientific: Do not give generic answers. Explain in detail the mechanisms of action, experimental data, or limitations stated in the document.
4. If the information in the snippets is insufficient to answer the question, state so frankly.

Context Snippets:
{snippets}

Question: {question}
Answer:
"""
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
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling Groq in answer_pdf_question: {e}. Falling back...")
            return answer_pdf_question_heuristic(chunks, question)
    else:
        url = _gemini_gen_url(api_key)
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
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"Error calling Gemini in answer_pdf_question: {e}. Falling back...")
            return answer_pdf_question_heuristic(chunks, question)

import time

def call_llm_with_retry(prompt, api_key, use_groq=False, max_retries=3, timeout_sec=120):
    """
    Generic function to call an LLM with retries.
    Uses either Groq or Gemini based on the use_groq flag or api_key prefix.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        
    for attempt in range(max_retries):
        try:
            if use_groq or api_key.startswith("gsk_"):
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
                    "Authorization": f"Bearer {api_key}"
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                    res_data = json.loads(response.read().decode('utf-8'))
                    return res_data['choices'][0]['message']['content']
            else:
                url = _gemini_gen_url(api_key)
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                }
                headers = {"Content-Type": "application/json"}
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                    res_data = json.loads(response.read().decode('utf-8'))
                    return res_data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error calling LLM after {max_retries} attempts: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            logger.info("Loading spaCy NER model (en_core_web_sm)...")
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy model not loaded: {e}")
            return None
    return _nlp

def extract_technical_entities(text: str):
    """
    Combines statistical NER with rule-based Noun Chunk extraction
    to find algorithms, datasets, metrics, and organizations.
    """
    nlp = get_nlp()
    if not nlp:
        return {"algorithms": [], "datasets": [], "metrics": [], "organizations": []}
    
    doc = nlp(text)
    
    entities = defaultdict(set)
    
    # 1. Statistical NER for Organizations/Institutions
    for ent in doc.ents:
        # Filter out common false positives and very short strings
        if ent.label_ == "ORG" and len(ent.text) > 2:
            clean_text = ent.text.replace('\n', ' ').strip()
            entities["organizations"].add(clean_text)
            
    # 2. Rule-based Noun Chunking for Technical Terms
    algorithm_keywords = {"model", "algorithm", "network", "architecture", "framework", "system", "transformer", "method"}
    dataset_keywords = {"dataset", "corpus", "data", "benchmark", "set"}
    metric_keywords = {"metric", "score", "accuracy", "precision", "recall", "f1", "loss", "error", "rate"}
    
    for chunk in doc.noun_chunks:
        head_lemma = chunk.root.lemma_.lower()
        chunk_text = chunk.text.replace('\n', ' ').strip()
        
        # Filter out pronouns and overly long/short chunks
        if len(chunk_text) < 4 or len(chunk_text.split()) > 4 or chunk.root.pos_ == "PRON":
            continue
            
        if head_lemma in algorithm_keywords:
            entities["algorithms"].add(chunk_text)
        elif head_lemma in dataset_keywords:
            entities["datasets"].add(chunk_text)
        elif head_lemma in metric_keywords:
            entities["metrics"].add(chunk_text)
            
    # Convert sets to sorted lists for deterministic output, taking top 5 of each
    return {
        "algorithms": sorted(list(entities["algorithms"]))[:5],
        "datasets": sorted(list(entities["datasets"]))[:5],
        "metrics": sorted(list(entities["metrics"]))[:5],
        "organizations": sorted(list(entities["organizations"]))[:5]
    }

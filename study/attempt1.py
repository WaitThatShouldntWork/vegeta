# %% [markdown]
# # Bayesian Active Inference Decider
# This is aBayesian Active Inference decider that chooses among Answer / Ask / Search 
# using Expected Information Gain (EIG) and bayesian brain theory.
# The intent behind this system is to be able to handle unclear queries and ultimately ask good questions back.

# %%
#TEST_UTTERANCE = "I'm looking for a spy movie with Pierce Brosnan from the 1990s"
#TEST_UTTERANCE = "What Bond film did Daniel Craig star in that won awards?
#TEST_UTTERANCE = "Recommend me a film" #combination of 20 questions and recommendation
TEST_UTTERANCE = "I'm thinking of a film. Try to guess it." # 20 questions game
#TEST_UTTERANCE = "I want a recommendation for action movies similar to Heat" # recommendation
#TEST_UTTERANCE = "Do this for me" # purposefully vague

# Multi-turn conversation settings
ENABLE_MULTITURN = True  # Set to True to use session management
SESSION_ID = None  # Will be set if continuing a session

# %% [markdown]
# Setup & Imports - Database connection, LLM client, basic utilities
import json
import numpy as np
import requests
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
import logging

# Import multi-turn session manager
try:
    from multiturn_session_manager import MultiTurnSessionManager
    MULTITURN_AVAILABLE = True
except ImportError:
    logger.warning("MultiTurnSessionManager not available - running single-turn only")
    MULTITURN_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
# DEFAULT_MODEL = "Gemma3:12b"
DEFAULT_MODEL = "qwen3:8b"

# System Parameters
DEFAULTS = {
    'K_anchors': 10,
    'M_candidates': 20,
    'hops': 2,
    'tau_retrieval': 0.7,
    'tau_posterior': 0.7,
    'alpha': 1.0,
    'beta': 0.5,
    'gamma': 0.3,
    'sigma_sem_sq': 0.3,
    'sigma_struct_sq': 0.2,
    'sigma_terms_sq': 0.2,
    'N_terms_max': 15,
    'N_expected': 20,
    'small_set_threshold': 3,
    'small_set_blend': 0.5,
    'lambda_missing': 0.30,
    'd_cap': 40,
    'lambda_hub': 0.02
}


print(f"Test utterance: '{TEST_UTTERANCE}'")

# Initialize session manager for multi-turn conversations
session_manager = None
conversation_context = {}

if ENABLE_MULTITURN and MULTITURN_AVAILABLE:
    session_manager = MultiTurnSessionManager()
    if SESSION_ID:
        # Continue existing session
        session = session_manager.get_session(SESSION_ID)
        if session:
            conversation_context = session_manager.get_conversation_context(SESSION_ID)
            print(f"Continuing session {SESSION_ID}, turn {len(session.turns)+1}")
        else:
            print(f"Session {SESSION_ID} not found or expired")
            SESSION_ID = None
    
    if not SESSION_ID:
        # Start new session
        SESSION_ID = session_manager.start_session()
        print(f"Started new session: {SESSION_ID}")

# %%
# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Test connection
try:
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("RETURN 'Connection successful' AS status")
        record = result.single()
        logger.info(f"Neo4j: {record['status']}")
except Exception as e:
    logger.error(f"Neo4j connection failed: {e}")

# Test Ollama connection
try:
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    if response.status_code == 200:
        models = [model['name'] for model in response.json().get('models', [])]
        logger.info(f"Ollama models available: {models}")
        if DEFAULT_MODEL not in models:
            logger.warning(f"Default model {DEFAULT_MODEL} not found. Available: {models}")
    else:
        logger.error(f"Ollama connection failed: HTTP {response.status_code}")
except Exception as e:
    logger.error(f"Ollama connection failed: {e}")

print("Setup complete!")
print(f"Neo4j URI: {NEO4J_URI}")
print(f"Ollama URL: {OLLAMA_BASE_URL}")
print(f"Default model: {DEFAULT_MODEL}")

# %% [markdown]
# # Entity Extraction - LLM-based term/entity extraction from utterances
# Extract canonical terms, entities, numbers, dates from user input using LLM in JSON mode

# %%
def call_ollama_json(prompt: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Call Ollama with JSON format enforcement"""
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent extraction
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response'])
    except Exception as e:
        logger.error(f"Ollama JSON call failed: {e}")
        return {}

def analyze_query_clarity(utterance: str) -> Dict[str, Any]:
    """Analyze query clarity and specificity using LLM"""
    
    analysis_prompt = f"""Analyze this user query for clarity and specificity. Return valid JSON only:

{{
    "clarity": "clear|moderate|vague|extremely_vague",
    "specificity": "specific|general|abstract", 
    "domain_identifiable": true/false,
    "immediate_clarification_needed": true/false,
    "clarification_type": "task_domain|intent|specifics|none",
    "reasoning": "brief explanation of the assessment"
}}

Rules for analysis:
- clarity: How understandable is the request?
  - clear: Specific, actionable request
  - moderate: Somewhat unclear but processable  
  - vague: Unclear what user wants
  - extremely_vague: No clear meaning (like "do this", "help me", "it")
- immediate_clarification_needed: true if you cannot proceed without asking for more info
- clarification_type: what type of clarification is most needed?

Query: "{utterance}"

JSON:"""
    
    try:
        result = call_ollama_json(analysis_prompt)
        return result
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        # Fallback analysis
        return {
            'clarity': 'moderate',
            'specificity': 'general',
            'domain_identifiable': False,
            'immediate_clarification_needed': False,
            'clarification_type': 'none',
            'reasoning': 'Analysis failed, using defaults'
        }

def extract_entities_llm(utterance: str) -> Dict[str, Any]:
    """Extract entities and terms using LLM with JSON schema validation and query analysis"""
    
    extraction_prompt = f"""Extract information from this user utterance. Return valid JSON only with this exact structure:

{{
    "canonical_terms": ["term1", "term2"],
    "entities": [
        {{"surface": "text_as_written", "normalized": "canonical_form", "type": "EntityType"}}
    ],
    "numbers": [1995, 2010],
    "dates": ["1995", "2010s"]
}}

Rules:
- canonical_terms: lowercase, lemmatized key terms (max 15)
- entities: extract any named entities mentioned (people, products, places, works, etc.)
- numbers: extract salient numeric values
- dates: extract date expressions

Utterance: "{utterance}"

JSON:"""
    
    try:
        result = call_ollama_json(extraction_prompt)
        
        # Validate and clean the result
        canonical_terms = result.get('canonical_terms', [])[:DEFAULTS['N_terms_max']]
        entities = result.get('entities', [])
        numbers = result.get('numbers', [])
        dates = result.get('dates', [])
        
        # Dedupe and clean canonical terms
        canonical_terms = list(dict.fromkeys([term.lower().strip() for term in canonical_terms if term.strip()]))
        
        # Add query analysis as a separate LLM call for reliability
        query_analysis = analyze_query_clarity(utterance)
        
        return {
            'canonical_terms': canonical_terms,
            'entities': entities,
            'numbers': numbers,
            'dates': dates,
            'query_analysis': query_analysis
        }
        
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        # Fallback to simple rule-based extraction
        return

# Test the extraction
extraction_result = extract_entities_llm(TEST_UTTERANCE)

print("Extraction test:")
print(f"Input: {TEST_UTTERANCE}")
print(f"Result: {json.dumps(extraction_result, indent=2)}")

# Create u_terms from extraction
u_terms_set = set(extraction_result['canonical_terms'])
print(f"\nu_terms_set: {u_terms_set}")

# %% [markdown]
# # Semantic Embedding - Generate embeddings for retrieval using Ollama

# %%
def get_embedding(text: str, model: str = "nomic-embed-text") -> np.ndarray:
    """Get embedding from Ollama"""
    payload = {
        "model": model,
        "prompt": text
    }
    
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return np.array(result['embedding'])
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return

def create_u_sem(utterance: str) -> np.ndarray:
    """Create semantic embedding u_sem for the utterance"""
    embedding = get_embedding(utterance)
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

def create_u_terms_vec(terms: List[str]) -> Optional[np.ndarray]:
    """Create term vector by averaging individual term embeddings"""
    if not terms:
        return None
    
    embeddings = []
    for term in terms:
        emb = get_embedding(term)
        embeddings.append(emb)
    
    if embeddings:
        # Average and normalize
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            return avg_embedding / norm
        return avg_embedding
    return None

# Test embeddings
print("Testing embeddings...")

# Create u_sem (main utterance embedding)
u_sem = create_u_sem(TEST_UTTERANCE)
print(f"u_sem shape: {u_sem.shape}")
print(f"u_sem norm: {np.linalg.norm(u_sem):.3f}")

# Create u_terms_vec if we have terms
if extraction_result['canonical_terms']:
    u_terms_vec = create_u_terms_vec(extraction_result['canonical_terms'])
    if u_terms_vec is not None:
        print(f"u_terms_vec shape: {u_terms_vec.shape}")
        print(f"u_terms_vec norm: {np.linalg.norm(u_terms_vec):.3f}")
    else:
        print("u_terms_vec: None (no terms)")
else:
    u_terms_vec = None
    print("No canonical terms extracted")

# Store for later use
observation_u = {
    'u_sem': u_sem,
    'u_terms_set': u_terms_set,
    'u_terms_vec': u_terms_vec,
    'u_meta': {
        'utterance': TEST_UTTERANCE,
        'extraction': extraction_result
    }
}
print(f"\nObservation u created with keys: {list(observation_u.keys())}")

# %% [markdown]
# # Graph Retrieval - Anchor selection and subgraph expansion
# Find relevant nodes and expand into candidate subgraphs using semantic similarity

# %%
def find_anchor_nodes(u_sem: np.ndarray, k: int = DEFAULTS['K_anchors']) -> List[Dict[str, Any]]:
    """Find top-K anchor nodes using semantic similarity"""
    
    # Query for entities with embeddings (assuming they exist in your graph)
    cypher_query = """
    MATCH (e:Entity)
    WHERE e.sem_emb IS NOT NULL
    RETURN e.id as id, e.name as name, e.sem_emb as sem_emb, labels(e) as labels
    LIMIT 100
    """
    
    anchors = []
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher_query)
            nodes = list(result)
            
            if not nodes:
                # Fallback: get nodes without embeddings and use name similarity
                fallback_query = """
                MATCH (e:Entity)
                RETURN e.id as id, e.name as name, labels(e) as labels
                LIMIT 50
                """
                result = session.run(fallback_query)
                nodes = list(result)
                
                # Simple text similarity as fallback
                for node in nodes[:k]:
                    anchors.append({
                        'id': node['id'],
                        'name': node['name'],
                        'labels': node['labels'],
                        's_sem': 0.5,  # Default similarity
                        's_graph': 0.5,
                        's_combined': 0.5
                    })
                
                logger.warning(f"No embeddings found, using {len(anchors)} nodes with default scores")
                return anchors
            
            # Calculate semantic similarities
            for node in nodes:
                if node['sem_emb']:
                    try:
                        node_emb = np.array(node['sem_emb'])
                        # L2 normalize
                        node_emb = node_emb / np.linalg.norm(node_emb)
                        
                        # Cosine similarity
                        s_sem = np.dot(u_sem, node_emb)
                        s_graph = s_sem  # For now, use same as semantic (would use graph embeddings if available)
                        
                        anchors.append({
                            'id': node['id'],
                            'name': node['name'],
                            'labels': node['labels'],
                            's_sem': float(s_sem),
                            's_graph': float(s_graph),
                            's_combined': float(0.7 * s_sem + 0.3 * s_graph)
                        })
                    except Exception as e:
                        logger.error(f"Error processing node {node['id']}: {e}")
            
    except Exception as e:
        logger.error(f"Anchor retrieval failed: {e}")
        return []
    
    # Sort by combined score and take top-K
    anchors.sort(key=lambda x: x['s_combined'], reverse=True)
    return anchors[:k]

def expand_subgraphs(anchors: List[Dict], hops: int = DEFAULTS['hops']) -> List[Dict[str, Any]]:
    """Expand anchors into candidate subgraphs"""
    
    candidates = []
    
    for i, anchor in enumerate(anchors):
        try:
            # Simple k-hop expansion
            cypher_query = f"""
            MATCH path = (start:Entity {{id: $anchor_id}})-[*1..{hops}]-(connected)
            WHERE connected:Entity OR connected:SlotValue
            WITH start, connected, path
            RETURN start.id as anchor_id,
                   collect(DISTINCT connected.id) as connected_ids,
                   collect(DISTINCT connected.name) as connected_names,
                   collect(DISTINCT labels(connected)) as connected_labels,
                   count(DISTINCT connected) as subgraph_size
            """
            
            with driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(cypher_query, anchor_id=anchor['id'])
                record = result.single()
                
                if record and record['connected_ids']:
                    # Create subgraph candidate
                    candidate = {
                        'id': f"subgraph_{i}",
                        'anchor_id': anchor['id'],
                        'anchor_name': anchor['name'],
                        'connected_ids': record['connected_ids'],
                        'connected_names': record['connected_names'],
                        'subgraph_size': record['subgraph_size'],
                        'anchor_score': anchor['s_combined'],
                        'retrieval_score': anchor['s_combined']  # Initial score from retrieval
                    }
                    candidates.append(candidate)
                
        except Exception as e:
            logger.error(f"Subgraph expansion failed for anchor {anchor['id']}: {e}")
    
    return candidates

def link_entities_to_graph(entities: List[Dict]) -> List[str]:
    """Link extracted entities to graph nodes using full-text search"""
    
    linked_ids = []
    
    for entity in entities:
        try:
            # Simple name matching (would use full-text index in production)
            cypher_query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($entity_name)
               OR any(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($entity_name))
            RETURN e.id as id, e.name as name
            LIMIT 3
            """
            
            with driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(cypher_query, entity_name=entity['normalized'])
                matches = list(result)
                
                for match in matches:
                    linked_ids.append(match['id'])
                    logger.info(f"Linked '{entity['surface']}' -> {match['name']} ({match['id']})")
                
        except Exception as e:
            logger.error(f"Entity linking failed for {entity}: {e}")
    
    return linked_ids

# New helpers: active checklist target labels, candidate enumeration, and neighbors
def get_active_checklist_and_target_labels() -> Tuple[Optional[str], List[str]]:
    """Determine the active checklist and its primary target labels in a generic way.

    Strategy (domain-agnostic):
    - Prefer a checklist that has at least one required SlotSpec with expect_labels defined.
    - Prefer SlotSpecs with cardinality ONE (they typically denote the primary target).
    - If multiple match, pick the first arbitrarily (we're not baking in domain rules).
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            cypher = """
            MATCH (ss:SlotSpec)
            RETURN ss.checklist_name AS checklist_name,
                   ss.name AS name,
                   ss.expect_labels AS expect_labels,
                   ss.required AS required,
                   ss.cardinality AS cardinality
            ORDER BY required DESC
            """
            records = list(session.run(cypher))
            # Filter to those with expect_labels
            candidates: List[Dict[str, Any]] = []
            for r in records:
                expect_labels = r.get("expect_labels") or []
                if expect_labels:
                    candidates.append({
                        "checklist_name": r.get("checklist_name"),
                        "name": r.get("name"),
                        "expect_labels": expect_labels,
                        "required": bool(r.get("required")),
                        "cardinality": r.get("cardinality") or "ONE"
                    })
            if not candidates:
                return None, []

            # Prefer required and cardinality ONE
            def sort_key(x: Dict[str, Any]):
                return (
                    1 if x["required"] else 0,
                    1 if str(x["cardinality"]).upper() == "ONE" else 0,
                    -len(x["expect_labels"])  # fewer labels preferred for specificity
                )
            candidates.sort(key=sort_key, reverse=True)
            picked = candidates[0]
            return picked["checklist_name"], picked["expect_labels"]
    except Exception as e:
        logger.error(f"Failed to fetch checklist target labels: {e}")
        return None, []

def enumerate_target_candidates_from_anchors(anchors: List[Dict[str, Any]], target_labels: List[str],
                                             hops: int = DEFAULTS['hops'], decay: float = 0.8,
                                             limit_per_anchor: int = 50) -> List[Dict[str, Any]]:
    """Enumerate candidate entities of the target labels within k hops of the anchors.

    - Score each candidate by max over anchors of (anchor_score * decay^(distance-1)).
    - Returns a list of candidate dicts with id, entity_id, name, labels, retrieval_score, best_anchor info.
    """
    if not anchors or not target_labels:
        return []

    # Map candidate_id -> best info
    candidate_map: Dict[str, Dict[str, Any]] = {}

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for anchor in anchors:
                anchor_id = anchor.get('id')
                anchor_name = anchor.get('name')
                anchor_score = float(anchor.get('s_combined', 0.0))
                if not anchor_id:
                    continue

                # Use variable-length paths and min(length) to avoid shortestPath start=end issues
                cypher = f"""
                MATCH (start:Entity {{id: $anchor_id}})
                MATCH p = (start)-[*1..{hops}]-(cand)
                WHERE cand.id <> start.id
                  AND any(lbl IN labels(cand) WHERE lbl IN $target_labels)
                WITH cand, min(length(p)) AS dist
                RETURN cand.id AS id, cand.name AS name, labels(cand) AS labels, dist
                ORDER BY dist ASC
                LIMIT $limit
                """
                results = session.run(cypher, anchor_id=anchor_id, target_labels=target_labels, limit=limit_per_anchor)
                for rec in results:
                    cand_id = rec["id"]
                    cand_name = rec["name"]
                    cand_labels = rec["labels"] or []
                    dist = max(1, int(rec["dist"]))
                    # Path-decayed score from this anchor
                    score = anchor_score * (decay ** (dist - 1))

                    existing = candidate_map.get(cand_id)
                    if (existing is None) or (score > existing.get('retrieval_score', 0.0)):
                        candidate_map[cand_id] = {
                            'entity_id': cand_id,
                            'name': cand_name,
                            'labels': cand_labels,
                            'retrieval_score': float(score),
                            'best_anchor': {
                                'id': anchor_id,
                                'name': anchor_name,
                                'dist': dist,
                                'anchor_score': anchor_score
                            }
                        }
    except Exception as e:
        logger.error(f"Failed to enumerate target candidates: {e}")
        return []

    # Convert to list and sort by retrieval_score
    candidates = sorted(candidate_map.values(), key=lambda x: x['retrieval_score'], reverse=True)
    return candidates

def get_candidate_neighbors(candidate_id: str) -> Tuple[List[str], List[str]]:
    """Return 1-hop neighbor ids and names for a candidate entity."""
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            cypher = """
            MATCH (c:Entity {id: $id})-[r]-(n)
            RETURN collect(DISTINCT n.id) AS ids, collect(DISTINCT n.name) AS names
            """
            rec = session.run(cypher, id=candidate_id).single()
            if rec:
                return rec.get('ids') or [], rec.get('names') or []
    except Exception as e:
        logger.error(f"Failed to fetch neighbors for {candidate_id}: {e}")
    return [], []

def compute_generic_term_evidence(candidate_id: str, canonical_terms: List[str]) -> Dict[str, Any]:
    """Compute generic evidence by matching canonical terms to SlotValue slots and Fact predicate names.

    This is domain-agnostic: it checks lowercase substring matches.
    Returns a dict with a normalized score in [0,1].
    """
    terms = [t.lower() for t in (canonical_terms or []) if isinstance(t, str) and t.strip()]
    if not terms:
        return {"term_evidence_score": 0.0, "matches": {}}

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            cypher = """
            MATCH (c:Entity {id: $id})
            OPTIONAL MATCH (c)-[:HAS_SLOT]->(sv:SlotValue)
            OPTIONAL MATCH (f1:Fact)-[:SUBJECT]->(c)
            OPTIONAL MATCH (f2:Fact)-[:OBJECT]->(c)
            OPTIONAL MATCH (f1)-[:PREDICATE]->(rt1:RelationType)
            OPTIONAL MATCH (f2)-[:PREDICATE]->(rt2:RelationType)
            RETURN collect(DISTINCT toLower(sv.slot)) AS slots,
                   collect(DISTINCT toLower(rt1.name)) AS preds1,
                   collect(DISTINCT toLower(rt2.name)) AS preds2
            """
            rec = session.run(cypher, id=candidate_id).single()
            slots: List[str] = rec.get('slots') or []
            preds1: List[str] = rec.get('preds1') or []
            preds2: List[str] = rec.get('preds2') or []
            preds = list(dict.fromkeys([p for p in (preds1 + preds2) if p]))
            matches: Dict[str, List[str]] = {}
            hit_count = 0
            for t in terms:
                matched = []
                if any(t in (s or '') for s in slots):
                    matched.append('slot')
                if any(t in (p or '') for p in preds):
                    matched.append('predicate')
                if matched:
                    matches[t] = matched
                    hit_count += 1
            score = hit_count / max(len(terms), 1)
            return {"term_evidence_score": float(score), "matches": matches}
    except Exception as e:
        logger.error(f"Failed to compute term evidence for {candidate_id}: {e}")
        return {"term_evidence_score": 0.0, "matches": {}}

# Execute retrieval (updated: target candidate enumeration)
print("Starting graph retrieval...")

# Find anchor nodes using semantic similarity
anchors = find_anchor_nodes(u_sem)
print(f"Found {len(anchors)} anchor nodes:")
for anchor in anchors[:3]:  # Show top 3
    print(f"  {anchor['name']} ({anchor['id']}) - score: {anchor['s_combined']:.3f}")

# Link extracted entities to graph
linked_entity_ids = link_entities_to_graph(extraction_result['entities'])
print(f"\nLinked entities: {linked_entity_ids}")

# Determine active checklist target labels
active_checklist_name, target_labels = get_active_checklist_and_target_labels()
if not target_labels:
    logger.warning("No target labels found from checklist; falling back to subgraph expansion.")
    candidates = expand_subgraphs(anchors)
else:
    # Enumerate target candidates near anchors
    target_candidates = enumerate_target_candidates_from_anchors(anchors, target_labels)
    # Attach neighbor info (ids + names) for feature generation
    candidates = []
    for i, cand in enumerate(target_candidates):
        neighbor_ids, neighbor_names = get_candidate_neighbors(cand['entity_id'])
        candidates.append({
            'id': f"cand_{i}",
            'anchor_id': cand['best_anchor']['id'],
            'anchor_name': cand['best_anchor']['name'],
            'connected_ids': neighbor_ids,
            'connected_names': neighbor_names,
            'subgraph_size': len(neighbor_ids),
            'anchor_score': cand['best_anchor']['anchor_score'],
            'retrieval_score': cand['retrieval_score'],
            'entity_id': cand['entity_id'],
            'entity_name': cand['name'],
            'entity_labels': cand['labels']
        })

    # Fallbacks if no target candidates found
    if not candidates:
        # 1) Use anchors that already match target labels
        label_matched = [a for a in anchors if any(lbl in (a.get('labels') or []) for lbl in target_labels)]
        for j, a in enumerate(label_matched):
            neighbor_ids, neighbor_names = get_candidate_neighbors(a['id'])
            candidates.append({
                'id': f"anchor_cand_{j}",
                'anchor_id': a['id'],
                'anchor_name': a['name'],
                'connected_ids': neighbor_ids,
                'connected_names': neighbor_names,
                'subgraph_size': len(neighbor_ids),
                'anchor_score': a.get('s_combined', 0.0),
                'retrieval_score': a.get('s_combined', 0.0),
                'entity_id': a['id'],
                'entity_name': a['name'],
                'entity_labels': a.get('labels') or []
            })

    if not candidates:
        # 2) Fall back to original subgraph expansion, adapted into candidate shape
        fallback = expand_subgraphs(anchors)
        for k, fc in enumerate(fallback):
            candidates.append({
                'id': f"fallback_{k}",
                'anchor_id': fc['anchor_id'],
                'anchor_name': fc['anchor_name'],
                'connected_ids': fc.get('connected_ids', []),
                'connected_names': fc.get('connected_names', []),
                'subgraph_size': fc.get('subgraph_size', 0),
                'anchor_score': fc.get('anchor_score', 0.0),
                'retrieval_score': fc.get('retrieval_score', 0.0),
                'entity_id': fc.get('anchor_id'),
                'entity_name': fc.get('anchor_name'),
                'entity_labels': []
            })

print(f"\nGenerated {len(candidates)} candidate targets:")
for i, candidate in enumerate(candidates[:3]):  # Show top 3
    shown_name = candidate.get('entity_name') or candidate.get('anchor_name')
    print(f"  Candidate {i}: target={shown_name}, size={candidate['subgraph_size']}, score={candidate['retrieval_score']:.3f}")

# Store retrieval context R
retrieval_context_R = {
    'anchors': anchors,
    'linked_entity_ids': linked_entity_ids,
    'candidates': candidates,
    'expansion_params': {'hops': DEFAULTS['hops'], 'k_anchors': DEFAULTS['K_anchors']},
    'utterance': TEST_UTTERANCE,
    'active_checklist': active_checklist_name,
    'target_labels': target_labels
}

print(f"\nRetrieval context R created with {len(candidates)} candidates")

# %% [markdown]
# # Feature Generation - Create observed and predicted features for likelihood
# Generate u_struct_obs and expected features u' for each candidate

# %%
def compute_u_struct_obs(candidate: Dict[str, Any]) -> Dict[str, int]:
    """Compute observed structural features for a candidate-centered neighborhood.

    Generic behavior:
    - Count neighbor label frequencies (excluding Demo) and edge types around candidate entity_id if present;
      otherwise fall back to original anchor/connected_ids logic.
    """
    
    try:
        if candidate.get('entity_id'):
            # Candidate-centric
            cypher_query = """
            MATCH (c:Entity {id: $cid})
            OPTIONAL MATCH (c)-[r]-(n)
            RETURN 
                count(DISTINCT n) as node_count,
                count(DISTINCT type(r)) as edge_type_count,
                collect(DISTINCT labels(n)) as node_label_groups,
                collect(DISTINCT type(r)) as edge_types
            """
            params = {"cid": candidate['entity_id']}
        else:
            # Fallback to anchor-based computation
            cypher_query = """
            MATCH (anchor:Entity {id: $anchor_id})
            OPTIONAL MATCH (anchor)-[r]-(connected)
            WHERE connected.id IN $connected_ids
            RETURN 
                count(DISTINCT connected) as node_count,
                count(DISTINCT type(r)) as edge_type_count,
                collect(DISTINCT labels(connected)) as node_label_groups,
                collect(DISTINCT type(r)) as edge_types
            """
            params = {"anchor_id": candidate['anchor_id'], "connected_ids": candidate['connected_ids']}
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher_query, **params)
            record = result.single()
            
            if record:
                # Flatten label groups and count each label
                all_labels = []
                for label_group in record['node_label_groups']:
                    if label_group:  # Skip None/empty groups
                        all_labels.extend(label_group)
                
                # Count label occurrences
                label_counts = {}
                for label in all_labels:
                    label_counts[f"label_{label}"] = label_counts.get(f"label_{label}", 0) + 1
                
                # Count edge types
                edge_counts = {}
                for edge_type in record['edge_types']:
                    if edge_type:
                        edge_counts[f"edge_{edge_type}"] = edge_counts.get(f"edge_{edge_type}", 0) + 1
                
                # Combine counts
                struct_features = {
                    'total_nodes': record['node_count'] or 0,
                    'total_edge_types': record['edge_type_count'] or 0,
                    **label_counts,
                    **edge_counts
                }
                
                return struct_features
            
    except Exception as e:
        logger.error(f"Failed to compute structural features for {candidate['id']}: {e}")
    
    # Fallback
    return {'total_nodes': candidate.get('subgraph_size', 0)}

def generate_expected_terms(candidate: Dict[str, Any]) -> List[str]:
    """Generate expected terms with a tighter, candidate-centric scope (domain-agnostic)."""
    
    expected_terms: List[str] = []

    # 1) Candidate entity name words
    if candidate.get('entity_name'):
        words = candidate['entity_name'].lower().replace('-', ' ').replace(':', ' ').split()
        expected_terms.extend([w for w in words if len(w) > 2])

    # 2) Direct SlotValue values for the candidate entity
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            if candidate.get('entity_id'):
                cypher_query = """
                MATCH (c:Entity {id: $cid})-[:HAS_SLOT]->(sv:SlotValue)
                RETURN sv.slot as slot, sv.value as value
                """
                params = {"cid": candidate['entity_id']}
            else:
                cypher_query = """
                MATCH (anchor:Entity {id: $anchor_id})-[:HAS_SLOT]->(sv:SlotValue)
                RETURN sv.slot as slot, sv.value as value
                """
                params = {"anchor_id": candidate['anchor_id']}

            result = session.run(cypher_query, **params)
            for record in result:
                if record['value']:
                    slot_words = str(record['value']).lower().split()
                    expected_terms.extend([w for w in slot_words if len(w) > 2])
                if record['slot']:
                    expected_terms.append(str(record['slot']).lower())
    
    except Exception as e:
        logger.error(f"Failed to get slot values for {candidate['anchor_id']}: {e}")
    
    # 3) Optionally a small sample of 1-hop neighbor names (kept small to avoid noise)
    for name in (candidate.get('connected_names') or [])[:3]:
        if name:
            words = name.lower().replace('-', ' ').replace(':', ' ').split()
            expected_terms.extend([w for w in words if len(w) > 2])

    # Dedupe and limit
    expected_terms = list(dict.fromkeys(expected_terms))[:DEFAULTS['N_expected']]
    return expected_terms

def compute_delta_distances(observation_u: Dict, candidate: Dict, expected_terms: List[str]) -> Dict[str, float]:
    """Compute delta distances for semantic, structural, and terms channels.

    Structural distance is checklist-driven (generic):
    - If active checklist and its SlotSpecs are available, reduce distance when
      expected labels are present around the candidate; increase when absent.
    - Otherwise fall back to a mild default.
    """
    
    # δ_sem: semantic distance (placeholder - would use actual subgraph embedding)
    # For now, use inverse of retrieval score as a proxy
    delta_sem = max(0.0, 1.0 - candidate.get('retrieval_score', 0.0))
    
    # δ_struct: checklist-driven
    delta_struct = 0.3
    try:
        active_checklist = retrieval_context_R.get('active_checklist')
        if active_checklist and candidate.get('u_struct_obs'):
            struct_obs = candidate['u_struct_obs']
            with driver.session(database=NEO4J_DATABASE) as session:
                cypher = """
                MATCH (ss:SlotSpec {checklist_name: $cl})
                RETURN ss.name AS name, ss.expect_labels AS expect_labels, ss.required AS required
                """
                specs = [r.data() for r in session.run(cypher, cl=active_checklist)]
            # Compute penalties for missing expected labels; rewards for present
            missing_pen = 0.0
            present_bonus = 0.0
            for spec in specs:
                labels = spec.get('expect_labels') or []
                if not labels:
                    continue
                has_any = any(struct_obs.get(f"label_{lbl}", 0) > 0 for lbl in labels)
                if has_any:
                    present_bonus += 0.05
                else:
                    if bool(spec.get('required')):
                        missing_pen += 0.15
                    else:
                        missing_pen += 0.05
            # Map to distance with floor/ceiling
            delta_struct = np.clip(0.3 + missing_pen - present_bonus, 0.0, 1.0)
    except Exception:
        pass
    
    # δ_terms: terms distance using Jaccard
    u_terms_set = observation_u.get('u_terms_set', set())
    expected_terms_set = set(expected_terms)
    
    if len(u_terms_set) >= DEFAULTS['small_set_threshold'] and len(expected_terms_set) >= DEFAULTS['small_set_threshold']:
        # Use Jaccard distance
        intersection = len(u_terms_set.intersection(expected_terms_set))
        union = len(u_terms_set.union(expected_terms_set))
        jaccard = intersection / union if union > 0 else 0.0
        delta_terms = 1.0 - jaccard
    else:
        # Use blended approach for small sets
        intersection = len(u_terms_set.intersection(expected_terms_set))
        union = len(u_terms_set.union(expected_terms_set))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Cosine similarity component (would use actual embeddings)
        cosine_sim = 0.3  # Placeholder
        
        delta_terms = DEFAULTS['small_set_blend'] * (1.0 - jaccard) + (1.0 - DEFAULTS['small_set_blend']) * (1.0 - cosine_sim)
    
    return {
        'delta_sem': delta_sem,
        'delta_struct': delta_struct,
        'delta_terms': delta_terms
    }

# Process all candidates to generate features
print("Generating features for candidates...")

for i, candidate in enumerate(candidates):
    # Compute observed structural features
    u_struct_obs = compute_u_struct_obs(candidate)
    candidate['u_struct_obs'] = u_struct_obs
    
    # Generate expected terms
    expected_terms = generate_expected_terms(candidate)
    candidate['expected_terms'] = expected_terms
    
    # Compute distances
    distances = compute_delta_distances(observation_u, candidate, expected_terms)
    candidate['distances'] = distances
    
    # Compute preliminary likelihood score
    alpha, beta, gamma = DEFAULTS['alpha'], DEFAULTS['beta'], DEFAULTS['gamma']
    sigma_sem_sq, sigma_struct_sq, sigma_terms_sq = DEFAULTS['sigma_sem_sq'], DEFAULTS['sigma_struct_sq'], DEFAULTS['sigma_terms_sq']
    
    # Log-likelihood (negative because we're computing negative log-likelihood)
    log_likelihood = -(
        alpha * distances['delta_sem'] / sigma_sem_sq +
        beta * distances['delta_struct'] / sigma_struct_sq +
        gamma * distances['delta_terms'] / sigma_terms_sq
    )
    
    candidate['log_likelihood'] = log_likelihood

# Show results for top candidates
print("\nFeature generation complete. Top candidates:")
candidates_sorted = sorted(candidates, key=lambda x: x['log_likelihood'], reverse=True)

for i, candidate in enumerate(candidates_sorted[:3]):
    print(f"\nCandidate {i}: {candidate['anchor_name']}")
    print(f"  Structure: {candidate['u_struct_obs']}")
    print(f"  Expected terms: {candidate['expected_terms'][:5]}...")  # Show first 5
    print(f"  Distances: δ_sem={candidate['distances']['delta_sem']:.3f}, δ_struct={candidate['distances']['delta_struct']:.3f}, δ_terms={candidate['distances']['delta_terms']:.3f}")
    print(f"  Log-likelihood: {candidate['log_likelihood']:.3f}")

# Update retrieval context with processed candidates
retrieval_context_R['candidates'] = candidates_sorted
print(f"\nUpdated retrieval context with {len(candidates_sorted)} processed candidates")

# %% [markdown]
# # Prior Construction - Build priors over hidden states
# Construct priors p(z_checklist), p(z_goal), p(z_subgraph), etc.

# %%
def build_checklist_prior(observation_u: Dict, dialogue_act_prior: Dict) -> Dict[str, float]:
    """Build prior over checklists based on dialogue acts and extracted terms"""
    
    # Query available checklists
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("MATCH (c:Checklist) RETURN c.name as name, c.description as description")
            checklists = list(result)
    except Exception as e:
        logger.error(f"Failed to query checklists: {e}")
        checklists = []
    
    if not checklists:
        # Fallback to generic checklists (domain-agnostic)
        checklists = [
            {'name': 'Identify', 'description': 'Identify a specific target from clues'},
            {'name': 'Recommend', 'description': 'Recommend items based on preferences'},
            {'name': 'Verify', 'description': 'Verify facts or properties'}
        ]
    
    # Extract signals from observation and dialogue acts
    canonical_terms = observation_u.get('u_meta', {}).get('extraction', {}).get('canonical_terms', [])
    dialogue_acts = dialogue_act_prior or {}
    
    # Intent-driven checklist selection
    checklist_prior = {}
    total_mass = 0.95  # Leave 5% for "none-of-the-above"
    
    # Strong signal: "recommendation" in canonical terms
    has_recommend_term = any('recommend' in term.lower() for term in canonical_terms)
    has_similar_pattern = any(word in ' '.join(canonical_terms).lower() for word in ['similar', 'like', 'comparable'])
    request_score = dialogue_acts.get('request', 0.0)
    
    logger.info(f"Intent signals: recommend_term={has_recommend_term}, similar_pattern={has_similar_pattern}, request_score={request_score:.3f}")
    
    # For most interactions, operate without rigid procedural constraints
    # Only use formal checklists when they exist and are clearly relevant
    if checklists:
        # Apply loose heuristic matching to existing formal checklists
        for checklist in checklists:
            name = checklist['name']
            description = checklist.get('description', '').lower()
            
            # Heuristic matching - don't force it
            if ('identify' in name.lower() or 'find' in description):
                if has_recommend_term:
                    checklist_prior[name] = 0.1  # Low weight when intent doesn't match
                else:
                    checklist_prior[name] = 0.4  # Moderate weight
            elif 'recommend' in name.lower() or 'suggest' in description:
                if has_recommend_term:
                    checklist_prior[name] = 0.4  # Moderate weight 
                else:
                    checklist_prior[name] = 0.1  # Low weight
            else:
                checklist_prior[name] = 0.2  # Neutral weight for unknown checklists
        
        logger.info(f"Matched {len(checklists)} formal checklists with flexible weights")
    
    # Store the detected intent for decision-making (independent of checklists)
    if has_recommend_term or has_similar_pattern:
        logger.info("Detected recommendation intent - will route to SEARCH in decision phase")
    elif request_score > 0.4:
        logger.info("Detected request intent - decision phase will determine best action")
    else:
        logger.info("Detected general inquiry - will use standard slot-based approach")
    
    # Always prioritize flexible operation over rigid procedures
    if checklist_prior:
        # Normalize existing checklists but keep them secondary to flexible operation
        total_formal_mass = 0.3  # Only 30% weight to formal checklists
        current_sum = sum(checklist_prior.values())
        if current_sum > 0:
            for name in checklist_prior:
                checklist_prior[name] = (checklist_prior[name] / current_sum) * total_formal_mass
    
    # High probability of operating without rigid procedural constraints
    checklist_prior['None'] = 0.7  # 70% chance of flexible, intent-driven operation
    
    return checklist_prior

def classify_user_intent_llm(utterance: str, dialogue_acts: Dict[str, float], extraction: Dict) -> Dict[str, float]:
    """Use LLM to understand user's underlying goals and intentions"""
    
    # Extract context for intent analysis
    canonical_terms = extraction.get('canonical_terms', [])
    entities = extraction.get('entities', [])
    primary_dialogue_act = max(dialogue_acts.items(), key=lambda x: x[1])[0]
    
    analysis_prompt = f"""Analyze this user utterance to understand their underlying goals. Return valid JSON only:

{{
    "user_goals": {{
        "identify": 0.0,
        "recommend": 0.0,
        "verify": 0.0,
        "explore": 0.0,
        "act": 0.0
    }},
    "primary_goal": "identify|recommend|verify|explore|act",
    "reasoning": "brief explanation of intent analysis"
}}

Goal definitions:
- identify: User wants to find/name a specific thing ("What movie is this?", "Who directed X?")
- recommend: User wants suggestions/recommendations ("Suggest movies like X", "What should I watch?")
- verify: User wants to confirm/check facts ("Is this true?", "Did X win an award?")
- explore: User wants to learn/understand more ("Tell me about X", "How does Y work?")
- act: User wants something done/executed ("Play this movie", "Add to watchlist")

Context:
- Utterance: "{utterance}"
- Primary dialogue act: {primary_dialogue_act}
- Key terms: {canonical_terms}
- Entities mentioned: {[e.get('surface', '') for e in entities]}

Guidelines:
- Focus on what the user ultimately wants to accomplish
- Consider specific language patterns (e.g., "like", "similar" → recommend)
- Dialogue acts provide hints but goals are deeper intentions
- Return probabilities that sum to 1.0

JSON:"""
    
    try:
        result = call_ollama_json(analysis_prompt)
        user_goals = result.get('user_goals', {})
        
        # Validate and normalize probabilities
        total = sum(user_goals.values())
        if total > 0:
            user_goals = {k: v/total for k, v in user_goals.items()}
        else:
            # Fallback based on dialogue act patterns
            if primary_dialogue_act == 'request':
                user_goals = {'identify': 0.4, 'recommend': 0.4, 'verify': 0.1, 'explore': 0.1, 'act': 0.0}
            elif primary_dialogue_act == 'clarify':
                user_goals = {'identify': 0.3, 'recommend': 0.1, 'verify': 0.2, 'explore': 0.4, 'act': 0.0}
            else:
                user_goals = {'identify': 0.4, 'recommend': 0.2, 'verify': 0.15, 'explore': 0.15, 'act': 0.1}
        
        return user_goals
        
    except Exception as e:
        logger.error(f"User intent classification failed: {e}")
        # Fallback to dialogue-act based heuristic
        if primary_dialogue_act == 'request':
            return {'identify': 0.4, 'recommend': 0.4, 'verify': 0.1, 'explore': 0.1, 'act': 0.0}
        elif primary_dialogue_act == 'clarify':
            return {'identify': 0.3, 'recommend': 0.1, 'verify': 0.2, 'explore': 0.4, 'act': 0.0}
        else:
            return {'identify': 0.4, 'recommend': 0.2, 'verify': 0.15, 'explore': 0.15, 'act': 0.1}

def build_goal_prior(checklist_prior: Dict[str, float]) -> Dict[str, float]:
    """Build prior over goals using LLM-based intent understanding"""
    # This function now serves as a bridge - the actual intent analysis happens 
    # in the main decision flow where we have access to the utterance and context
    
    # For now, return a neutral prior that will be overridden by LLM analysis
    return {'identify': 0.4, 'recommend': 0.2, 'verify': 0.15, 'explore': 0.15, 'act': 0.1}

def build_subgraph_prior(candidates: List[Dict]) -> Dict[str, float]:
    """Build prior over candidate subgraphs using retrieval scores and simplicity"""
    
    if not candidates:
        return {}
    
    # Use retrieval scores as base, add simplicity bias
    subgraph_prior = {}
    
    for candidate in candidates:
        cand_id = candidate['id']
        
        # Base score from retrieval
        retrieval_score = candidate.get('retrieval_score', 0.0)
        
        # Simplicity bias (prefer smaller subgraphs)
        size = candidate.get('subgraph_size', 1)
        simplicity_bonus = 1.0 / (1.0 + 0.1 * size)  # Gentle penalty for larger graphs
        
        # Provenance bonus (placeholder)
        provenance_bonus = 1.0  # Would factor in source reliability
        
        # Recency bonus (placeholder)
        recency_bonus = 1.0     # Would factor in recent mentions
        
        # Combined prior score
        prior_score = retrieval_score * simplicity_bonus * provenance_bonus * recency_bonus
        subgraph_prior[cand_id] = prior_score
    
    # Apply softmax with temperature for normalization
    tau = DEFAULTS['tau_retrieval']
    
    # Softmax normalization
    max_score = max(subgraph_prior.values()) if subgraph_prior else 0
    exp_scores = {k: np.exp((v - max_score) / tau) for k, v in subgraph_prior.items()}
    total_exp = sum(exp_scores.values())
    
    if total_exp > 0:
        subgraph_prior = {k: v / total_exp for k, v in exp_scores.items()}
    
    return subgraph_prior

def classify_dialogue_act_llm(utterance: str) -> Dict[str, float]:
    """Classify dialogue act using LLM understanding of conversational intent"""
    
    analysis_prompt = f"""Analyze this utterance and classify the dialogue act. Return valid JSON only:

{{
    "dialogue_acts": {{
        "clarify": 0.0,
        "confirm": 0.0, 
        "request": 0.0,
        "provide": 0.0
    }},
    "primary_act": "clarify|confirm|request|provide",
    "reasoning": "brief explanation of classification"
}}

Dialogue acts defined:
- clarify: Asking for information, seeking explanation (e.g., "What is X?", "How does Y work?")
- confirm: Seeking verification of information (e.g., "Is this correct?", "Are you sure?")  
- request: Asking for something to be done (e.g., "Find me X", "I want Y", "Help me with Z")
- provide: Giving information or stating facts (e.g., "X is Y", "I think Z")

Return probabilities that sum to 1.0 across the four dialogue acts.

Utterance: "{utterance}"

JSON:"""
    
    try:
        result = call_ollama_json(analysis_prompt)
        dialogue_acts = result.get('dialogue_acts', {})
        
        # Validate and normalize probabilities
        total = sum(dialogue_acts.values())
        if total > 0:
            dialogue_acts = {k: v/total for k, v in dialogue_acts.items()}
        else:
            # Fallback to uniform distribution
            dialogue_acts = {'clarify': 0.25, 'confirm': 0.25, 'request': 0.25, 'provide': 0.25}
            
        return dialogue_acts
        
    except Exception as e:
        logger.error(f"Dialogue act classification failed: {e}")
        # Fallback to uniform distribution
        return {'clarify': 0.25, 'confirm': 0.25, 'request': 0.25, 'provide': 0.25}

def build_dialogue_act_prior(utterance: str) -> Dict[str, float]:
    """Build prior over dialogue acts using LLM-based classification"""
    return classify_dialogue_act_llm(utterance)

def build_novelty_prior(anchors: List[Dict], observation_u: Dict) -> float:
    """Build prior over novelty/OOD signals"""
    
    # Check semantic similarity to anchors
    if anchors:
        max_anchor_sim = max(anchor.get('s_combined', 0.0) for anchor in anchors)
    else:
        max_anchor_sim = 0.0
    
    # Novelty indicators
    novelty_signals = []
    
    # Low similarity to all anchors
    if max_anchor_sim < 0.35:
        novelty_signals.append(0.3)
    
    # Few extracted entities linked to graph
    extraction = observation_u.get('u_meta', {}).get('extraction', {})
    entities = extraction.get('entities', [])
    if len(entities) > 0:
        linked_ratio = len(retrieval_context_R.get('linked_entity_ids', [])) / len(entities)
        if linked_ratio < 0.3:
            novelty_signals.append(0.2)
    
    # Short utterance (hard to interpret)
    utterance = observation_u.get('u_meta', {}).get('utterance', '')
    if len(utterance.split()) < 5:
        novelty_signals.append(0.1)
    
    # Combine novelty signals
    novelty_score = min(0.8, sum(novelty_signals))  # Cap at 0.8
    
    return novelty_score

# Build all priors (with multi-turn conversation history)
print("Building priors over hidden states...")

# Check if we have conversation history to inform priors
if session_manager and SESSION_ID:
    session_state = session_manager.get_session(SESSION_ID)
    if session_state and session_state.turns:
        # Build priors from conversation history
        print("Using conversation history to inform priors...")
        history_informed_priors = session_manager.build_priors_from_history(session_state)
        
        # Use history-informed priors as base, then adjust with current utterance
        dialogue_act_prior = history_informed_priors.get('dialogue_act', {})
        checklist_prior = history_informed_priors.get('checklist', {})
        goal_prior = history_informed_priors.get('goal', {})
        
        # Update with current utterance information
        current_dialogue_acts = build_dialogue_act_prior(TEST_UTTERANCE)
        current_checklist = build_checklist_prior(observation_u, current_dialogue_acts)
        current_goal = classify_user_intent_llm(
            utterance=TEST_UTTERANCE,
            dialogue_acts=current_dialogue_acts, 
            extraction=extraction_result
        )
        
        # Blend history and current (70% history, 30% current for continuity)
        blend_weight = 0.7
        
        # Blend dialogue acts
        for key in set(dialogue_act_prior.keys()) | set(current_dialogue_acts.keys()):
            hist_val = dialogue_act_prior.get(key, 0.0)
            curr_val = current_dialogue_acts.get(key, 0.0)
            dialogue_act_prior[key] = blend_weight * hist_val + (1 - blend_weight) * curr_val
        
        # Blend checklist
        for key in set(checklist_prior.keys()) | set(current_checklist.keys()):
            hist_val = checklist_prior.get(key, 0.0)
            curr_val = current_checklist.get(key, 0.0)
            checklist_prior[key] = blend_weight * hist_val + (1 - blend_weight) * curr_val
        
        # Blend goal  
        for key in set(goal_prior.keys()) | set(current_goal.keys()):
            hist_val = goal_prior.get(key, 0.0)
            curr_val = current_goal.get(key, 0.0)
            goal_prior[key] = blend_weight * hist_val + (1 - blend_weight) * curr_val
        
        # Normalize blended priors
        dialogue_act_total = sum(dialogue_act_prior.values())
        if dialogue_act_total > 0:
            dialogue_act_prior = {k: v/dialogue_act_total for k, v in dialogue_act_prior.items()}
        
        checklist_total = sum(checklist_prior.values())
        if checklist_total > 0:
            checklist_prior = {k: v/checklist_total for k, v in checklist_prior.items()}
            
        goal_total = sum(goal_prior.values())
        if goal_total > 0:
            goal_prior = {k: v/goal_total for k, v in goal_prior.items()}
        
        print("Blended history-informed priors with current utterance")
    else:
        print("No conversation history available, using fresh priors")
        # Fresh conversation - build priors normally
        dialogue_act_prior = build_dialogue_act_prior(TEST_UTTERANCE)
        checklist_prior = build_checklist_prior(observation_u, dialogue_act_prior)
        goal_prior = classify_user_intent_llm(
            utterance=TEST_UTTERANCE,
            dialogue_acts=dialogue_act_prior, 
            extraction=extraction_result
        )
else:
    print("Single-turn mode - building fresh priors")
    # Single-turn mode - build priors normally
    dialogue_act_prior = build_dialogue_act_prior(TEST_UTTERANCE)
    checklist_prior = build_checklist_prior(observation_u, dialogue_act_prior)
    goal_prior = classify_user_intent_llm(
        utterance=TEST_UTTERANCE,
        dialogue_acts=dialogue_act_prior, 
        extraction=extraction_result
    )

print(f"Dialogue act prior: {dialogue_act_prior}")
print(f"Checklist prior: {checklist_prior}")
print(f"Goal prior (LLM-based): {goal_prior}")

# Subgraph prior (enhanced with conversation history hints if available)
subgraph_prior = build_subgraph_prior(retrieval_context_R['candidates'])

if session_manager and SESSION_ID:
    session_state = session_manager.get_session(SESSION_ID)
    if session_state and session_state.turns:
        history_subgraph_hints = session_manager._build_subgraph_prior_from_history(session_state, {})
        # Boost subgraph candidates that were recently mentioned with high confidence
        for cand_id, boost in history_subgraph_hints.items():
            if cand_id in subgraph_prior:
                subgraph_prior[cand_id] *= (1.0 + boost)
        
        # Renormalize
        total = sum(subgraph_prior.values())
        if total > 0:
            subgraph_prior = {k: v/total for k, v in subgraph_prior.items()}

print(f"Subgraph prior (top 3): {dict(list(subgraph_prior.items())[:3])}")

# Novelty prior (adapted based on conversation history)
base_novelty = build_novelty_prior(retrieval_context_R['anchors'], observation_u)
if session_manager and SESSION_ID:
    session_state = session_manager.get_session(SESSION_ID)
    if session_state:
        novelty_prior = session_manager._adapt_novelty_prior(session_state, base_novelty)
    else:
        novelty_prior = base_novelty
else:
    novelty_prior = base_novelty

print(f"Novelty prior: {novelty_prior:.3f}")

# Store all priors
priors = {
    'checklist': checklist_prior,
    'goal': goal_prior,
    'subgraph': subgraph_prior,
    'dialogue_act': dialogue_act_prior,
    'novelty': novelty_prior
}

print(f"\nPriors constructed for {len(priors)} hidden variables")

# %% [markdown]
# # Likelihood & Posterior Update - Variational inference to update beliefs
# Compute p(u|v) and update posterior q(v) using Bayesian inference

# %%
def compute_likelihood_per_candidate(candidate: Dict, observation_u: Dict, priors: Dict) -> float:
    """Compute full likelihood p(u|v) for a candidate including all factors"""
    
    # We already computed the channel-based likelihood in feature generation
    channel_log_likelihood = candidate['log_likelihood']
    
    # Add penalty terms for missing required slots and hub nodes
    penalties = 0.0
    
    # Missing required slot penalty (schema-driven): check required SlotSpec for top checklist
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            top_checklist = max(priors.get('checklist', {"None": 1.0}).items(), key=lambda x: x[1])[0]
            cypher = """
            MATCH (ss:SlotSpec {checklist_name: $cl}) WHERE ss.required = true
            RETURN ss.name AS name, ss.expect_labels AS expect_labels
            """
            records = session.run(cypher, cl=top_checklist)
            required_specs = [r.data() for r in records]
    except Exception:
        required_specs = []
    
    struct_obs = candidate.get('u_struct_obs', {})
    for spec in required_specs:
        expected_labels = spec.get('expect_labels') or []
        # If any expected label count is missing or zero, penalize
        if expected_labels and not any(struct_obs.get(f"label_{lbl}", 0) > 0 for lbl in expected_labels):
            penalties += DEFAULTS['lambda_missing']
    
    # Hub node penalty
    subgraph_size = candidate.get('subgraph_size', 0)
    if subgraph_size > DEFAULTS['d_cap']:
        hub_penalty = DEFAULTS['lambda_hub'] * (subgraph_size - DEFAULTS['d_cap'])
        penalties += hub_penalty
    
    # Generic evidence bonus from SlotValues and Facts matching canonical terms
    try:
        canon_terms = list(observation_u.get('u_meta', {}).get('extraction', {}).get('canonical_terms', []))
        if candidate.get('entity_id') and canon_terms:
            ev = compute_generic_term_evidence(candidate['entity_id'], canon_terms)
            # Convert [0,1] score to a small positive bonus; temperature controls impact
            bonus = 0.6 * ev.get('term_evidence_score', 0.0)
            channel_log_likelihood += bonus
    except Exception:
        pass

    # Total log-likelihood
    # Minor adjustment using observation_u terms overlap
    try:
        u_terms_set = observation_u.get('u_terms_set', set())
        expected_terms = set(candidate.get('expected_terms', []))
        if u_terms_set and expected_terms:
            jaccard = len(u_terms_set.intersection(expected_terms)) / max(1, len(u_terms_set.union(expected_terms)))
            channel_log_likelihood += 0.2 * jaccard
    except Exception:
        pass
    
    total_log_likelihood = channel_log_likelihood - penalties
    
    return total_log_likelihood

def update_posterior_subgraph(candidates: List[Dict], priors: Dict, observation_u: Dict) -> Dict[str, float]:
    """Update posterior over subgraphs using Bayesian inference"""
    
    # Compute posterior ∝ p(u|v) * p(v) for each candidate
    posterior_scores = {}
    
    for candidate in candidates:
        cand_id = candidate['id']
        
        # Likelihood p(u|v)
        log_likelihood = compute_likelihood_per_candidate(candidate, observation_u, priors)
        
        # Prior p(v)
        prior_prob = priors['subgraph'].get(cand_id, 1e-6)  # Small floor to avoid log(0)
        log_prior = np.log(prior_prob)
        
        # Posterior score (in log space)
        log_posterior = log_likelihood + log_prior
        posterior_scores[cand_id] = log_posterior
        
        # Store in candidate for tracking
        candidate['log_likelihood_full'] = log_likelihood
        candidate['log_prior'] = log_prior
        candidate['log_posterior'] = log_posterior
    
    # Normalize using softmax with temperature
    tau = DEFAULTS['tau_posterior']
    max_score = max(posterior_scores.values()) if posterior_scores else 0
    
    exp_scores = {k: np.exp((v - max_score) / tau) for k, v in posterior_scores.items()}
    total_exp = sum(exp_scores.values())
    
    if total_exp > 0:
        posterior_subgraph = {k: v / total_exp for k, v in exp_scores.items()}
    else:
        # Fallback to uniform
        n = len(posterior_scores)
        posterior_subgraph = {k: 1.0/n for k in posterior_scores.keys()} if n > 0 else {}
    
    return posterior_subgraph

def update_posterior_checklist(observation_u: Dict, priors: Dict, top_subgraph_candidate: Dict) -> Dict[str, float]:
    """Update checklist posterior based on evidence"""
    
    # For simplicity, adjust checklist prior based on subgraph evidence
    checklist_posterior = priors['checklist'].copy()
    
    # Use generic signals: if we observe concentrated evidence for a specific type, prefer Identify-like checklists
    struct_obs = top_subgraph_candidate.get('u_struct_obs', {})
    has_specific_entity = any(k.startswith('label_') and v > 0 for k, v in struct_obs.items())
    
    if has_specific_entity:
        for key in checklist_posterior:
            if 'Identify' in key:
                checklist_posterior[key] *= 1.2
    
    # If evidence suggests multiple related entities, slightly boost Recommend-like
    total_labels = sum(v for k, v in struct_obs.items() if k.startswith('label_'))
    if total_labels >= 2:
        for key in checklist_posterior:
            if 'Recommend' in key:
                checklist_posterior[key] *= 1.1
    
    # Normalize
    total = sum(checklist_posterior.values())
    if total > 0:
        checklist_posterior = {k: v/total for k, v in checklist_posterior.items()}
    
    return checklist_posterior

def update_posterior_goal(observation_u: Dict, priors: Dict, dialogue_act_posterior: Dict) -> Dict[str, float]:
    """Update goal posterior - simplified since LLM already does sophisticated intent analysis"""
    
    # Since LLM-based goal priors already incorporate dialogue acts, entities, and context,
    # we can trust them more and apply minimal adjustments
    goal_posterior = priors['goal'].copy()
    
    # Optional: minor adjustments based on evidence strength (future enhancement)
    # For now, trust the LLM analysis
    
    return goal_posterior

# Perform posterior updates
print("Updating posterior beliefs...")

# Update subgraph posterior (main inference)
posterior_subgraph = update_posterior_subgraph(retrieval_context_R['candidates'], priors, observation_u)

if not posterior_subgraph:
    logger.warning("No posterior candidates found; using first retrieval candidate if available.")
    if retrieval_context_R['candidates']:
        top_candidate = retrieval_context_R['candidates'][0]
        top_subgraph_id = top_candidate['id']
        posterior_subgraph = {top_subgraph_id: 1.0}
    else:
        raise RuntimeError("No candidates available for posterior update.")
else:
    # Get top candidate for conditioning other posteriors
    top_subgraph_id = max(posterior_subgraph.items(), key=lambda x: x[1])[0]
    top_candidate = next(c for c in retrieval_context_R['candidates'] if c['id'] == top_subgraph_id)

shown_name = top_candidate.get('entity_name') or top_candidate.get('anchor_name')
print(f"Top candidate: {shown_name} (probability: {posterior_subgraph[top_subgraph_id]:.3f})")

# Update other posteriors
posterior_checklist = update_posterior_checklist(observation_u, priors, top_candidate)
posterior_goal = update_posterior_goal(observation_u, priors, priors['dialogue_act'])  # Using prior as proxy

# Dialogue act posterior (simplified - just use prior for now)
posterior_dialogue_act = priors['dialogue_act'].copy()

# Store all posteriors
posteriors = {
    'checklist': posterior_checklist,
    'goal': posterior_goal,
    'subgraph': posterior_subgraph,
    'dialogue_act': posterior_dialogue_act,
    'novelty': priors['novelty']  # Static for now
}

# Show results
print("\nPosterior updates complete:")
print(f"Checklist posterior: {posterior_checklist}")
print(f"Goal posterior: {dict(list(posterior_goal.items())[:3])}")  # Top 3
print(f"Subgraph posterior (top 3): {dict(list(posterior_subgraph.items())[:3])}")

# Compute confidence metrics
top_subgraph_prob = max(posterior_subgraph.values())
second_best_prob = sorted(posterior_subgraph.values(), reverse=True)[1] if len(posterior_subgraph) > 1 else 0
margin = top_subgraph_prob - second_best_prob

print(f"\nConfidence metrics:")
print(f"Top subgraph probability: {top_subgraph_prob:.3f}")
print(f"Margin (top1 - top2): {margin:.3f}")

# %% [markdown]
# # Uncertainty Quantification & Decision Policy
# Calculate entropy and EIG to decide between ANSWER, ASK, or SEARCH

# %%
def calculate_entropies(posteriors: Dict) -> Dict[str, float]:
    """Calculate Shannon entropy for each posterior distribution"""
    
    entropies = {}
    
    for var_name, distribution in posteriors.items():
        if isinstance(distribution, dict):
            # Shannon entropy: H = -Σ p(x) log p(x)
            entropy = 0.0
            for prob in distribution.values():
                if prob > 1e-10:  # Avoid log(0)
                    entropy -= prob * np.log2(prob)
            entropies[var_name] = entropy
        else:
            # Scalar value (e.g., novelty)
            entropies[var_name] = 0.0
    
    return entropies

def analyze_slot_uncertainty(top_candidate: Dict, checklist_name: str) -> Dict[str, Dict]:
    """Analyze uncertainty for required slots for the active checklist using SlotSpec."""
    
    slot_analysis: Dict[str, Dict] = {}
    struct_obs = top_candidate.get('u_struct_obs', {})
    
    # Fetch SlotSpec for this checklist from the graph
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            cypher = """
            MATCH (ss:SlotSpec {checklist_name: $cl})
            RETURN ss.name AS name, ss.required AS required, ss.expect_labels AS expect_labels, ss.cardinality AS cardinality
            ORDER BY ss.name
            """
            records = session.run(cypher, cl=checklist_name)
            slot_specs = [r.data() for r in records]
    except Exception:
        slot_specs = []
    
    for spec in slot_specs:
        name = spec.get('name')
        expected_labels = spec.get('expect_labels') or []
        required = bool(spec.get('required'))
        
        info = {'entropy': 1.0, 'confidence': 0.0, 'evidence': 'missing', 'required': required}
        
        # Confidence based on presence of any expected label evidence
        has_evidence = any(struct_obs.get(f"label_{lbl}", 0) > 0 for lbl in expected_labels)
        if has_evidence:
            info['confidence'] = 0.8 if required else 0.6
            info['entropy'] = 0.3 if required else 0.5
            info['evidence'] = 'structural'
        else:
            # If we have related terms in expected_terms, treat as weak evidence
            expected_terms = top_candidate.get('expected_terms', [])
            if any(str(name).lower() in t for t in expected_terms):
                info['confidence'] = 0.3
                info['entropy'] = 1.0
                info['evidence'] = 'weak'
        
        slot_analysis[name] = info
    
    return slot_analysis

def assess_slot_information_value_llm(slot: str, slot_info: Dict, candidate: Dict, query_context: Dict) -> float:
    """Use LLM to assess the information value of asking about a specific slot"""
    
    # Extract candidate evidence details
    distances = candidate.get('distances', {})
    semantic_distance = distances.get('delta_sem', 1.0)
    structural_distance = distances.get('delta_struct', 1.0)
    terms_distance = distances.get('delta_terms', 1.0)
    
    analysis_prompt = f"""Assess the information value of asking about this slot. Return valid JSON only:

{{
    "information_value": 0.0,
    "priority": "high|medium|low",
    "reasoning": "brief explanation of assessment"
}}

Slot Analysis:
- Slot: {slot}
- Current entropy: {slot_info['entropy']:.2f} bits
- Current confidence: {slot_info['confidence']:.3f}
- Evidence quality: {slot_info['evidence']}
- Required for decision: {slot_info.get('required', False)}

Candidate Evidence:
- Semantic distance: {semantic_distance:.3f} (lower = better match)
- Structural distance: {structural_distance:.3f} (lower = better match) 
- Terms distance: {terms_distance:.3f} (lower = better match)
- Connected entities: {len(candidate.get('connected_ids', []))}

Query Context:
- Query type: {query_context.get('type', 'general')}
- Domain: {query_context.get('domain', 'general')}
- User goal: {query_context.get('goal', 'identify')}

Assessment Guidelines:
- information_value: Expected bits of information gained (0.0-3.0 range)
- High entropy + low confidence = high potential gain
- Missing evidence in critical slots = very high value
- Strong existing evidence = lower marginal value
- Consider whether this slot is essential for the user's goal

JSON:"""
    
    try:
        result = call_ollama_json(analysis_prompt)
        
        info_value = float(result.get('information_value', 0.0))
        # Clamp to reasonable range
        info_value = max(0.0, min(3.0, info_value))
        
        return info_value
        
    except Exception as e:
        logger.error(f"Slot information value assessment failed: {e}")
        
        # Intelligent fallback based on information theory
        base_eig = slot_info['entropy']
        confidence_factor = 1.0 - slot_info['confidence']
        
        # Smart evidence factor based on evidence type
        evidence_factor_map = {
            'missing': 1.3,      # High value - we need this info
            'weak': 1.1,         # Moderate value - could be helpful
            'structural': 0.9,   # Lower value - we have some structural evidence
            'strong': 0.4        # Low value - we already have good evidence
        }
        evidence_factor = evidence_factor_map.get(slot_info['evidence'], 1.0)
        
        return base_eig * confidence_factor * evidence_factor

def calculate_eig_ask(slot_analysis: Dict, candidate: Dict = None, query_context: Dict = None) -> Dict[str, float]:
    """Calculate Expected Information Gain for asking about each slot using LLM assessment"""
    
    eig_ask = {}
    
    # Default context if not provided
    if candidate is None:
        candidate = {}
    if query_context is None:
        query_context = {'type': 'general', 'domain': 'film', 'goal': 'identify'}
    
    for slot, info in slot_analysis.items():
        # Use LLM to assess information value
        eig_value = assess_slot_information_value_llm(slot, info, candidate, query_context)
        eig_ask[slot] = eig_value
    
    return eig_ask

def calculate_eig_search() -> float:
    """Calculate Expected Information Gain for search actions"""
    
    # Simple heuristic: search is valuable when we have candidates but low confidence
    # Would be more sophisticated in practice (considering specific facts to verify)
    
    base_search_eig = 0.5  # Default search value
    
    # Adjust based on novelty
    novelty = posteriors.get('novelty', 0.0)
    if novelty > 0.3:
        base_search_eig *= 1.5  # Search more valuable for novel queries
    
    return base_search_eig

def predict_user_responses_llm(question: str, context: Dict) -> List[Tuple[str, float]]:
    """Use LLM to predict how users would likely respond to a specific question"""
    
    # Extract context information
    domain = context.get('domain', 'general')
    user_knowledge = context.get('user_knowledge', 'moderate')
    conversation_state = context.get('conversation_state', 'initial')
    
    analysis_prompt = f"""Predict how users would likely respond to this question. Return valid JSON only:

{{
    "response_probabilities": [
        {{"type": "specific_value", "probability": 0.0, "example": "example response"}},
        {{"type": "range_value", "probability": 0.0, "example": "example response"}},
        {{"type": "approximate", "probability": 0.0, "example": "example response"}},
        {{"type": "additional_clues", "probability": 0.0, "example": "example response"}},
        {{"type": "redirect_question", "probability": 0.0, "example": "example response"}},
        {{"type": "dont_know", "probability": 0.0, "example": "example response"}},
        {{"type": "clarify_question", "probability": 0.0, "example": "example response"}}
    ],
    "reasoning": "brief explanation of prediction"
}}

Response types defined:
- specific_value: User gives exact, definitive answer (e.g., "1995", "Heat", "Michael Mann")
- range_value: User gives range or category (e.g., "mid-90s", "action movie", "around 2 hours")
- approximate: User gives approximate/uncertain answer (e.g., "I think it was...", "probably...")
- additional_clues: User provides related information instead (e.g., "It had Tom Cruise", "It was about a heist")
- redirect_question: User asks back or changes topic (e.g., "Why do you ask?", "What about X instead?")
- dont_know: User admits they don't know (e.g., "I don't know", "Not sure", "No idea")
- clarify_question: User asks for clarification (e.g., "Which film?", "What do you mean?")

Context:
- Domain: {domain}
- User knowledge level: {user_knowledge}
- Conversation state: {conversation_state}

Question: "{question}"

Return probabilities that sum to 1.0.

JSON:"""
    
    try:
        result = call_ollama_json(analysis_prompt)
        response_probs = result.get('response_probabilities', [])
        
        # Convert to the expected format and normalize
        responses = []
        total_prob = 0.0
        
        for resp in response_probs:
            prob = float(resp.get('probability', 0.0))
            resp_type = resp.get('type', 'unknown')
            responses.append((resp_type, prob))
            total_prob += prob
        
        # Normalize probabilities
        if total_prob > 0:
            responses = [(resp_type, prob/total_prob) for resp_type, prob in responses]
        else:
            # Fallback to uniform distribution
            n = len(responses) if responses else 7
            default_prob = 1.0 / n
            responses = [("specific_value", default_prob), ("approximate", default_prob), 
                        ("dont_know", default_prob), ("additional_clues", default_prob),
                        ("range_value", default_prob), ("redirect_question", default_prob),
                        ("clarify_question", default_prob)]
        
        return responses

    except Exception as e:
        logger.error(f"User response prediction failed: {e}")
        # Fallback to simple heuristic
        return [("specific_value", 0.4), ("approximate", 0.3), ("dont_know", 0.2), ("additional_clues", 0.1)]

def simulate_user_responses(target_slot: str, posteriors: Dict) -> List[Tuple[str, float]]:
    """Simulate possible user responses using LLM-based prediction"""
    
    # Generate an appropriate question for this slot
    question = f"Could you tell me about the {target_slot}?"
    
    # Build context for prediction
    context = {
        'domain': 'movies',  # Could be extracted from the graph/checklist
        'user_knowledge': 'moderate',  # Could be inferred from previous interactions
        'conversation_state': 'clarification'  # Current state of conversation
    }
    
    return predict_user_responses_llm(question, context)

def assess_information_gain_llm(response_type: str, target_slot: str, posteriors: Dict, question_context: Dict) -> Dict[str, float]:
    """Use LLM to intelligently assess both immediate and follow-up information gain"""
    
    # Calculate current state metrics
    current_entropy = -sum(p * np.log2(p) for p in posteriors['subgraph'].values() if p > 1e-10)
    num_candidates = len(posteriors['subgraph'])
    top_prob = max(posteriors['subgraph'].values())
    
    # Extract context about the question and domain
    question_target = question_context.get('target', target_slot)
    question_type = question_context.get('type', 'clarification')
    domain = question_context.get('domain', 'general')
    
    analysis_prompt = f"""Analyze the expected information gain from this user response type. Return valid JSON only:

{{
    "immediate_entropy_reduction": 0.0,
    "followup_information_potential": 0.0,
    "reasoning": "brief explanation of information theory assessment"
}}

Current Uncertainty State:
- Total candidates: {num_candidates}
- Current entropy: {current_entropy:.2f} bits
- Top candidate probability: {top_prob:.3f}
- Uncertainty level: {"high" if current_entropy > 2.5 else "moderate" if current_entropy > 1.5 else "low"}

Question Context:
- Target information: {question_target}
- Question type: {question_type}
- Domain: {domain}

User Response Type: {response_type}

Response Type Meanings:
- specific_value: User provides exact, definitive answer
- range_value: User provides range/category (e.g., "1990s", "action genre")
- approximate: User provides rough/uncertain answer
- additional_clues: User offers related but different information
- redirect_question: User asks counter-question or changes topic
- dont_know: User explicitly states lack of knowledge
- clarify_question: User asks for clarification of our question

Assessment Guidelines:
- immediate_entropy_reduction: How much uncertainty (0.0-1.0 fraction) this response would eliminate
- followup_information_potential: Expected information gain (0.0-1.0) from best follow-up action
- Consider domain characteristics (films have many attributes, facts are binary)
- Higher entropy states have more potential for large reductions
- Specific values in high-uncertainty contexts are extremely valuable
- "Don't know" responses may redirect strategy (SEARCH vs ASK)

JSON:"""
    
    try:
        result = call_ollama_json(analysis_prompt)
        
        immediate_reduction = float(result.get('immediate_entropy_reduction', 0.0))
        followup_potential = float(result.get('followup_information_potential', 0.0))
        reasoning = result.get('reasoning', 'LLM analysis completed')
        
        # Validate ranges
        immediate_reduction = max(0.0, min(1.0, immediate_reduction))
        followup_potential = max(0.0, min(1.0, followup_potential))
        
        return {
            'immediate_reduction': immediate_reduction,
            'followup_potential': followup_potential,
            'reasoning': reasoning
        }
        
    except Exception as e:
        logger.error(f"Information gain assessment failed: {e}")
        
        # Intelligent fallback based on response type patterns
        fallback_map = {
            "specific_value": {'immediate': 0.8, 'followup': 0.1},
            "range_value": {'immediate': 0.5, 'followup': 0.4}, 
            "approximate": {'immediate': 0.3, 'followup': 0.5},
            "additional_clues": {'immediate': 0.4, 'followup': 0.7},
            "redirect_question": {'immediate': 0.1, 'followup': 0.3},
            "dont_know": {'immediate': 0.0, 'followup': 0.6},
            "clarify_question": {'immediate': 0.2, 'followup': 0.4}
        }
        
        fallback = fallback_map.get(response_type, {'immediate': 0.3, 'followup': 0.4})
        return {
            'immediate_reduction': fallback['immediate'],
            'followup_potential': fallback['followup'],
            'reasoning': f'LLM failed, using fallback for {response_type}'
        }

def estimate_entropy_reduction(response_type: str, target_slot: str, posteriors: Dict) -> float:
    """Estimate immediate entropy reduction using LLM-based analysis"""
    
    question_context = {
        'target': target_slot,
        'type': 'clarification',
        'domain': 'film'  # Default to film domain for this system
    }
    
    assessment = assess_information_gain_llm(response_type, target_slot, posteriors, question_context)
    current_entropy = -sum(p * np.log2(p) for p in posteriors['subgraph'].values() if p > 1e-10)
    
    return current_entropy * assessment['immediate_reduction']

def estimate_followup_eig(response_type: str, target_slot: str, posteriors: Dict) -> float:
    """Estimate follow-up information gain using LLM-based analysis"""
    
    question_context = {
        'target': target_slot,
        'type': 'clarification', 
        'domain': 'film'  # Default to film domain for this system
    }
    
    assessment = assess_information_gain_llm(response_type, target_slot, posteriors, question_context)
    
    # Scale follow-up potential by baseline EIG (assume moderate baseline of 1.0 bits)
    baseline_eig = 1.0
    return baseline_eig * assessment['followup_potential']

def assess_evidence_quality_factors_llm(candidate: Dict) -> Dict[str, Any]:
    """Use LLM to assess evidence quality factors instead of hard-coded thresholds"""
    
    distances = candidate.get('distances', {})
    semantic_distance = distances.get('delta_sem', 1.0)
    structural_distance = distances.get('delta_struct', 1.0) 
    terms_distance = distances.get('delta_terms', 1.0)
    connected_entities = len(candidate.get('connected_ids', []))
    
    analysis_prompt = f"""Analyze evidence quality for this candidate. Return valid JSON only:

{{
    "semantic_match": "strong|moderate|weak",
    "structural_match": "strong|moderate|weak", 
    "term_match": "strong|moderate|weak",
    "overall_evidence": "strong|moderate|weak|insufficient",
    "reasoning": "brief explanation of evidence assessment"
}}

Evidence Metrics:
- Semantic distance: {semantic_distance:.3f} (0.0 = perfect match, 1.0 = no match)
- Structural distance: {structural_distance:.3f} (0.0 = perfect match, 1.0 = no match)
- Terms distance: {terms_distance:.3f} (0.0 = perfect match, 1.0 = no match)
- Connected entities: {connected_entities}

Assessment Guidelines:
- Semantic distances < 0.3 are typically strong matches
- Structural distances < 0.2 indicate good structural alignment  
- Terms distances < 0.4 suggest good terminology overlap
- More connected entities generally indicate richer context
- Consider the combination of all factors for overall assessment

JSON:"""
    
    try:
        result = call_ollama_json(analysis_prompt)
        
        # Convert qualitative assessments to boolean for compatibility
        quality_map = {'strong': True, 'moderate': True, 'weak': False, 'insufficient': False}
        
        return {
            'semantic_match': quality_map.get(result.get('semantic_match', 'weak'), False),
            'structural_match': quality_map.get(result.get('structural_match', 'weak'), False),
            'term_match': quality_map.get(result.get('term_match', 'weak'), False),
            'has_entities': connected_entities > 0,
            'overall_quality': result.get('overall_evidence', 'weak'),
            'reasoning': result.get('reasoning', 'Evidence quality assessment completed')
        }
        
    except Exception as e:
        logger.error(f"Evidence quality factors assessment failed: {e}")
        
        # Intelligent fallback using reasonable thresholds
        return {
            'semantic_match': semantic_distance < 0.4,
            'structural_match': structural_distance < 0.3,
            'term_match': terms_distance < 0.5,
            'has_entities': connected_entities > 0,
            'overall_quality': 'moderate' if (semantic_distance < 0.5 and structural_distance < 0.4) else 'weak',
            'reasoning': 'LLM assessment failed, using fallback thresholds'
        }

def assess_evidence_quality_llm(top_candidate: Dict, posteriors: Dict, entropies: Dict, margin: float) -> Dict[str, Any]:
    """Use LLM to assess if current evidence is sufficient for making a decision"""
    
    # Extract evidence summary
    top_prob = max(posteriors['subgraph'].values())
    second_prob = sorted(posteriors['subgraph'].values(), reverse=True)[1] if len(posteriors['subgraph']) > 1 else 0.0
    
    # Build evidence summary for LLM
    evidence_summary = {
        'top_candidate_name': top_candidate.get('entity_name') or top_candidate.get('anchor_name'),
        'top_probability': float(top_prob),
        'second_probability': float(second_prob),
        'margin': float(margin),
        'subgraph_entropy': float(entropies.get('subgraph', 0.0)),
        'evidence_types': assess_evidence_quality_factors_llm(top_candidate),
        'candidate_count': len(posteriors['subgraph'])
    }
    
    analysis_prompt = f"""Assess whether we have sufficient evidence to give a confident answer. Return valid JSON only:

{{
    "decision_ready": true/false,
    "confidence_sufficient": true/false,
    "margin_sufficient": true/false,
    "recommended_action": "answer|ask|search",
    "reasoning": "brief explanation of assessment"
}}

Evidence Assessment:
- Top candidate: {evidence_summary['top_candidate_name']}
- Top probability: {evidence_summary['top_probability']:.3f}
- Second probability: {evidence_summary['second_probability']:.3f}
- Margin between top 2: {evidence_summary['margin']:.3f}
- Total candidates: {evidence_summary['candidate_count']}
- Uncertainty (entropy): {evidence_summary['subgraph_entropy']:.3f}

Evidence Quality:
- Semantic match: {'Strong' if evidence_summary['evidence_types']['semantic_match'] else 'Weak'}
- Structural match: {'Strong' if evidence_summary['evidence_types']['structural_match'] else 'Weak'}
- Term match: {'Strong' if evidence_summary['evidence_types']['term_match'] else 'Weak'}
- Overall quality: {evidence_summary['evidence_types'].get('overall_quality', 'unknown')}
- Connected entities: {len(top_candidate.get('connected_ids', []))}

Guidelines:
- decision_ready: true if evidence strongly points to one answer
- confidence_sufficient: true if top probability indicates strong evidence
- margin_sufficient: true if there's clear separation between top candidates
- Consider both the strength of evidence AND the clarity of the winner

JSON:"""
    
    try:
        result = call_ollama_json(analysis_prompt)
        return {
            'decision_ready': result.get('decision_ready', False),
            'confidence_sufficient': result.get('confidence_sufficient', False), 
            'margin_sufficient': result.get('margin_sufficient', False),
            'recommended_action': result.get('recommended_action', 'ask'),
            'reasoning': result.get('reasoning', 'Evidence assessment failed'),
            'evidence_summary': evidence_summary
        }
    except Exception as e:
        logger.error(f"Evidence quality assessment failed: {e}")
        # Fallback to conservative thresholds
        return {
            'decision_ready': top_prob >= 0.70 and margin >= 0.20,
            'confidence_sufficient': top_prob >= 0.70,
            'margin_sufficient': margin >= 0.20, 
            'recommended_action': 'answer' if (top_prob >= 0.70 and margin >= 0.20) else 'ask',
            'reasoning': 'LLM assessment failed, using fallback thresholds',
            'evidence_summary': evidence_summary
        }

def calculate_eig_lookahead(target_slot: str, posteriors: Dict) -> float:
    """Calculate 2-step lookahead EIG: immediate + expected follow-up value"""
    
    # EIG_1: Immediate entropy reduction from asking this slot
    possible_responses = simulate_user_responses(target_slot, posteriors)
    
    immediate_eig = 0.0
    followup_eig = 0.0
    
    for response_type, prob in possible_responses:
        # Immediate information gain
        entropy_reduction = estimate_entropy_reduction(response_type, target_slot, posteriors)
        immediate_eig += prob * entropy_reduction
        
        # Expected follow-up value (discounted)
        followup_value = estimate_followup_eig(response_type, target_slot, posteriors)
        followup_eig += prob * followup_value
    
    # EIG_2: Total lookahead value with discount factor
    discount = 0.7  # Future actions worth 70% of immediate actions
    total_eig = immediate_eig + discount * followup_eig
    
    return total_eig

def make_decision(posteriors: Dict, entropies: Dict, top_candidate: Dict, margin: float, observation_u: Dict, retrieval_context_R: Dict) -> Dict[str, Any]:
    """Decide between ANSWER, ASK, or SEARCH based on uncertainty and lookahead EIG"""
    
    # LLM-BASED FAST PATH: Use the LLM's query analysis for immediate clarification
    extraction = observation_u.get('u_meta', {}).get('extraction', {})
    query_analysis = extraction.get('query_analysis', {})
    
    # Check if LLM determined immediate clarification is needed
    if query_analysis.get('immediate_clarification_needed', False):
        clarity = query_analysis.get('clarity', 'vague')
        clarification_type = query_analysis.get('clarification_type', 'task_domain')
        
        # Generate appropriate clarification based on LLM analysis
        clarification_responses = {
            'task_domain': "I'd be happy to help! Could you tell me what specific task you need assistance with?",
            'intent': "I'm having trouble understanding what you need. Could you provide more details about what you're trying to accomplish?",
            'specifics': "Could you give me more specific details about what you're looking for?"
        }
        
        quick_response = clarification_responses.get(clarification_type, 
                                                   "Could you provide more information about what you need?")
        
        # Bypass heavy computation when LLM identifies need for immediate clarification
        return {
            'action': 'ASK',
            'confidence': 0.0,  # We know nothing about the actual task
            'margin': 0.0,
            'reasoning': f'LLM analysis: {clarity} query needs {clarification_type} clarification',
            'target': clarification_type,
            'quick_response': quick_response,
            'llm_analysis': query_analysis,
            'eig_scores': {
                'ask_immediate': {clarification_type: float('inf')},  # Infinite value for LLM-identified necessity
                'ask_lookahead': {clarification_type: float('inf')},
                'search': 0.0  # No point searching when LLM says we need clarification first
            }
        }
    
    # LLM-based evidence assessment will replace hard-coded thresholds
    
    top_prob = max(posteriors['subgraph'].values())
    
    # Get active checklist
    top_checklist = max(posteriors['checklist'].items(), key=lambda x: x[1])[0]
    
    # EARLY ROUTING: Intent-based routing for recommendations (independent of formal checklists)
    
    # Extract intent signals directly from user observation
    canonical_terms = observation_u.get('u_meta', {}).get('extraction', {}).get('canonical_terms', [])
    has_recommend_term = any('recommend' in term.lower() for term in canonical_terms)
    has_similar_pattern = any(word in ' '.join(canonical_terms).lower() for word in ['similar', 'like', 'comparable'])
    
    # Check if this looks like a recommendation request regardless of checklist
    is_recommendation_intent = (
        has_recommend_term or 
        has_similar_pattern or
        'Recommend' in top_checklist
    )
    
    if (is_recommendation_intent and 
        retrieval_context_R.get('linked_entity_ids') and 
        top_prob < 0.95):  # Not already supremely confident in answer
        
        linked_entity = retrieval_context_R.get('linked_entity_ids')[0]
        logger.info(f"Early routing: Recommendation intent detected with linked entity {linked_entity}")
        return {
            'action': 'SEARCH',
            'confidence': top_prob,
            'margin': margin,
            'reasoning': f"Recommendation intent with known entity ({linked_entity}) - searching for similar items",
            'target': 'similar_items',
            'eig_scores': {
                'ask_immediate': {},
                'ask_lookahead': {},
                'search': 1.0  # High EIG for recommendation search
            }
        }
    
    # Analyze slots for non-recommendation tasks or when entity linking failed
    slot_analysis = analyze_slot_uncertainty(top_candidate, top_checklist)
    
    # Build query context for LLM assessment
    query_context = {
        'type': 'clarification',
        'domain': 'film',  # Default to film domain for this system
        'goal': max(posteriors['goal'].items(), key=lambda x: x[1])[0] if 'goal' in posteriors else 'identify'
    }
    
    eig_ask_immediate = calculate_eig_ask(slot_analysis, top_candidate, query_context)
    eig_search = calculate_eig_search()
    
    # Calculate lookahead EIG for each slot (EIG_2)
    eig_ask_lookahead = {}
    for slot in eig_ask_immediate.keys():
        eig_ask_lookahead[slot] = calculate_eig_lookahead(slot, posteriors)
    
    # Decision logic
    decision = {
        'action': None,
        'confidence': top_prob,
        'margin': margin,
        'reasoning': '',
        'target': None,
        'eig_scores': {
            'ask_immediate': eig_ask_immediate,
            'ask_lookahead': eig_ask_lookahead,
            'search': eig_search
        }
    }
    
    # Use LLM to assess evidence quality instead of hard-coded thresholds
    evidence_assessment = assess_evidence_quality_llm(top_candidate, posteriors, entropies, margin)
    
    # Check if we should ANSWER based on LLM assessment
    if evidence_assessment['decision_ready']:
        decision['action'] = 'ANSWER'
        decision['target'] = top_candidate.get('entity_name') or top_candidate['anchor_name']
        decision['reasoning'] = f"LLM assessment: {evidence_assessment['reasoning']}"
        decision['evidence_assessment'] = evidence_assessment
    
    else:
        # Choose between ASK and SEARCH using lookahead EIG
        
        # Find best slot using lookahead EIG
        if eig_ask_lookahead:
            best_slot = max(eig_ask_lookahead.items(), key=lambda x: x[1])
            best_ask_eig = best_slot[1]
            best_ask_slot = best_slot[0]
            immediate_eig = eig_ask_immediate.get(best_ask_slot, 0)
        else:
            best_ask_eig = 0
            best_ask_slot = None
            immediate_eig = 0
        
        # Compare lookahead ASK vs SEARCH
        if best_ask_eig > eig_search and best_ask_slot:
            decision['action'] = 'ASK'
            decision['target'] = best_ask_slot
            decision['reasoning'] = f"LLM assessment: insufficient evidence. Asking about '{best_ask_slot}' has highest lookahead EIG ({best_ask_eig:.3f})"
        else:
            decision['action'] = 'SEARCH'
            decision['target'] = 'missing_facts'
            decision['reasoning'] = f"LLM assessment: insufficient evidence. Search has higher EIG ({eig_search:.3f}) than asking"
        
        # Include evidence assessment in decision for transparency
        decision['evidence_assessment'] = evidence_assessment
    
    return decision

# Perform uncertainty analysis and decision making
print("Analyzing uncertainty and making decision...")

# Calculate entropies
entropies = calculate_entropies(posteriors)
print(f"Entropies: {entropies}")

# Make decision
decision = make_decision(posteriors, entropies, top_candidate, margin, observation_u, retrieval_context_R)

print(f"\nDECISION: {decision['action']}")
print(f"Target: {decision['target']}")
print(f"Reasoning: {decision['reasoning']}")
print(f"Confidence: {decision['confidence']:.3f}")
print(f"EIG scores:")
print(f"  ASK immediate: {decision['eig_scores']['ask_immediate']}")
print(f"  ASK lookahead: {decision['eig_scores']['ask_lookahead']}")
print(f"  SEARCH: {decision['eig_scores']['search']:.3f}")

# Store decision for final system state
system_state = {
    'observation_u': observation_u,
    'retrieval_context_R': retrieval_context_R,
    'priors': priors,
    'posteriors': posteriors,
    'entropies': entropies,
    'decision': decision,
    'top_candidate': top_candidate
}

print(f"\nSystem state captured. Ready for action execution.")

# %% [markdown]
# # Question Generation & End-to-End Test
# Generate natural language questions and demonstrate the complete system

# %%
def generate_question_llm(decision: Dict, system_state: Dict) -> str:
    """Generate a natural language question using LLM"""
    
    if decision['action'] != 'ASK':
        return f"Action: {decision['action']} (target: {decision['target']})"
    
    target_slot = decision['target']
    
    # Handle LLM-based clarification (intelligent instant response)
    if 'quick_response' in decision:
        return decision['quick_response']
    
    top_candidate = system_state['top_candidate']
    
    # Get active checklist to determine question context
    active_checklist = max(system_state.get('posteriors', {}).get('checklist', {"None": 1.0}).items(), key=lambda x: x[1])[0]
    
    # Build context for question generation based on task type
    if 'Recommend' in active_checklist:
        # For recommendation tasks, ask about preferences rather than identification
        task_context = "recommend similar items"
        reasoning = f"Need to understand user preferences for {target_slot} to improve recommendations"
        fallback_template = f"What {target_slot} preference do you have for recommendations?"
    else:
        # For identification tasks, ask about target characteristics  
        task_context = "identify the target"
        reasoning = f"Need to disambiguate {target_slot} to improve identification"
        fallback_template = f"Could you share the {target_slot} you're thinking of?"
    
    context = {
        "checklist": active_checklist,
        "task_context": task_context,
        "slot": target_slot,
        "top_candidate": {
            "anchor": top_candidate['anchor_name'],
            "connected": top_candidate.get('connected_names', [])[:3]
        },
        "confidence": decision['confidence'],
        "reasoning": reasoning
    }
    
    # Create prompt for question generation
    prompt = f"""Generate a natural, conversational question to ask the user. Context:

- Task: {context['task_context']} 
- Need to ask about: {context['slot']}
- Current focus: {context['top_candidate']['anchor']}
- Confidence: {context['confidence']:.1%}
- Why: {context['reasoning']}

Generate ONE short, natural question that would help reduce uncertainty. Be conversational and helpful.

Question:"""
    
    try:
        response = call_ollama_json(f"{prompt}\n\nReturn JSON with format: {{\"question\": \"your question here\"}}")
        if 'question' in response:
            return response['question']
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
    
    # Fallback to context-appropriate template
    return fallback_template

def generate_answer(decision: Dict, system_state: Dict) -> str:
    """Generate an answer when confidence is high enough"""
    
    if decision['action'] != 'ANSWER':
        return None
    
    top_candidate = system_state['top_candidate']
    confidence = decision['confidence']
    
    # Build answer from top candidate
    answer = f"Based on your query, I believe you're looking for information about {top_candidate['anchor_name']}"
    
    # Add a couple of connected names if available
    connected_names = top_candidate.get('connected_names', [])
    if connected_names:
        answer += f", related to {', '.join(connected_names[:2])}"
    
    answer += f". (Confidence: {confidence:.1%})"
    
    return answer

def demonstrate_full_system():
    """Demonstrate the complete system with detailed step-by-step logging"""
    
    print("=== COMPLETE SYSTEM EXECUTION SUMMARY ===\n")
    
    # 1. Input & Initial Setup
    print("1. INPUT & SETUP")
    print("-" * 30)
    print(f"Query: '{TEST_UTTERANCE}'")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Default model: {DEFAULT_MODEL}")
    print()
    
    # 2. Entity Extraction Results
    print("2. ENTITY EXTRACTION")
    print("-" * 30)
    print(f"Canonical terms: {extraction_result['canonical_terms']}")
    print(f"Entities: {extraction_result['entities']}")
    print(f"Numbers: {extraction_result['numbers']}")
    print(f"Dates: {extraction_result['dates']}")
    print(f"u_terms_set: {u_terms_set}")
    print()
    
    # 3. Semantic Embeddings
    print("3. SEMANTIC EMBEDDINGS")
    print("-" * 30)
    print(f"u_sem shape: {u_sem.shape}")
    print(f"u_sem norm: {np.linalg.norm(u_sem):.3f}")
    if u_terms_vec is not None:
        print(f"u_terms_vec shape: {u_terms_vec.shape}")
        print(f"u_terms_vec norm: {np.linalg.norm(u_terms_vec):.3f}")
    else:
        print("u_terms_vec: None")
    print()
    
    # 4. Graph Retrieval
    print("4. GRAPH RETRIEVAL")
    print("-" * 30)
    print(f"Found {len(anchors)} anchor nodes:")
    for i, anchor in enumerate(anchors[:3]):
        print(f"  {i+1}. {anchor['name']} ({anchor['id']}) - score: {anchor['s_combined']:.3f}")
    print(f"Linked entity IDs: {linked_entity_ids}")
    print(f"Generated {len(candidates)} candidate subgraphs:")
    for i, candidate in enumerate(candidates[:3]):
        print(f"  {i+1}. Anchor: {candidate['anchor_name']}, Size: {candidate['subgraph_size']}, Score: {candidate['retrieval_score']:.3f}")
    print()
    
    # 5. Feature Generation
    print("5. FEATURE GENERATION")
    print("-" * 30)
    candidates_sorted = sorted(candidates, key=lambda x: x['log_likelihood'], reverse=True)
    for i, candidate in enumerate(candidates_sorted[:3]):
        print(f"Candidate {i+1}: {candidate['anchor_name']}")
        print(f"  Structure: {candidate['u_struct_obs']}")
        print(f"  Expected terms: {candidate['expected_terms'][:5]}...")
        print(f"  Distances: δ_sem={candidate['distances']['delta_sem']:.3f}, δ_struct={candidate['distances']['delta_struct']:.3f}, δ_terms={candidate['distances']['delta_terms']:.3f}")
        print(f"  Log-likelihood: {candidate['log_likelihood']:.3f}")
    print()
    
    # 6. Prior Construction
    print("6. PRIOR CONSTRUCTION")
    print("-" * 30)
    print(f"Checklist prior: {priors['checklist']}")
    print(f"Goal prior: {priors['goal']}")
    print(f"Subgraph prior (top 3): {dict(list(priors['subgraph'].items())[:3])}")
    print(f"Dialogue act prior: {priors['dialogue_act']}")
    print(f"Novelty prior: {priors['novelty']:.3f}")
    print()
    
    # 7. Posterior Updates
    print("7. POSTERIOR UPDATES")
    print("-" * 30)
    shown_name = top_candidate.get('entity_name') or top_candidate.get('anchor_name')
    print(f"Top candidate: {shown_name} (probability: {posteriors['subgraph'][top_subgraph_id]:.3f})")
    print(f"Checklist posterior: {posteriors['checklist']}")
    print(f"Goal posterior: {dict(list(posteriors['goal'].items())[:3])}")
    print(f"Subgraph posterior (top 3): {dict(list(posteriors['subgraph'].items())[:3])}")
    print(f"Confidence metrics:")
    print(f"  Top subgraph probability: {max(posteriors['subgraph'].values()):.3f}")
    print(f"  Margin (top1 - top2): {margin:.3f}")
    print()
    
    # 8. Uncertainty Analysis & Decision
    print("8. UNCERTAINTY ANALYSIS & DECISION")
    print("-" * 30)
    print(f"Entropies: {entropies}")
    print(f"DECISION: {decision['action']}")
    print(f"Target: {decision['target']}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"Confidence: {decision['confidence']:.3f}")
    print(f"EIG scores:")
    print(f"  ASK immediate: {decision['eig_scores']['ask_immediate']}")
    print(f"  ASK lookahead: {decision['eig_scores']['ask_lookahead']}")
    print(f"  SEARCH: {decision['eig_scores']['search']:.3f}")
    print()
    
    # 9. Final Output
    print("9. FINAL OUTPUT")
    print("-" * 30)
    if decision['action'] == 'ASK':
        question = generate_question_llm(decision, system_state)
        print(f"SYSTEM QUESTION: {question}")
    elif decision['action'] == 'ANSWER':
        answer = generate_answer(decision, system_state)
        print(f"SYSTEM ANSWER: {answer}")
    else:
        print(f"SYSTEM ACTION: [Searching for additional information about {decision['target']}]")
    print(f"Final reasoning: {decision['reasoning']}")
    print(f"Final confidence: {decision['confidence']:.1%}")
    print()

# Execute final question generation and demonstration
if decision['action'] == 'ASK':
    final_question = generate_question_llm(decision, system_state)
    print(f"GENERATED QUESTION: {final_question}")
elif decision['action'] == 'ANSWER':
    final_answer = generate_answer(decision, system_state)
    print(f"GENERATED ANSWER: {final_answer}")
else:
    print(f"SEARCH ACTION: Looking for {decision['target']}")

print(f"\nFinal system reasoning: {decision['reasoning']}")

# Run demonstration
demonstrate_full_system()

print("\n" + "="*60)
print("BAYESIAN ACTIVE INFERENCE SYSTEM - COMPLETE")
print("="*60)
print(f"Query: '{TEST_UTTERANCE}'")
print(f"Action: {decision['action']} → {decision['target']}")
print(f"Top candidate: {system_state['top_candidate']['anchor_name']}")
print(f"Confidence: {decision['confidence']:.1%}")

# Save session state for multi-turn continuation
if session_manager and SESSION_ID:
    turn_result = {
        'user_utterance': TEST_UTTERANCE,
        'action': decision['action'],
        'target': decision['target'],
        'confidence': decision['confidence'],
        'entropy': entropies.get('subgraph', 0.0),
        'reasoning': decision['reasoning'],
        'posteriors': posteriors,
        'priors': priors,
        'top_candidates': retrieval_context_R['candidates'][:3],  # Top 3 candidates
        'response': final_question if decision['action'] == 'ASK' else 
                   (final_answer if decision['action'] == 'ANSWER' else f"Searching for {decision['target']}")
    }
    
    success = session_manager.update_session_state(SESSION_ID, turn_result)
    if success:
        print(f"Session state saved for session {SESSION_ID}")
        context = session_manager.get_conversation_context(SESSION_ID)
        print(f"Conversation summary: {context.get('conversation_summary', 'None')}")
        print(f"Total turns: {context.get('turn_count', 0)}")
    else:
        print("Failed to save session state")

print(f"System ready for next interaction!")


# %%
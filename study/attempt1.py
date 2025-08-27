# %% [markdown]
# # Bayesian Active Inference Decider


# %% [markdown]
# Test Configuration - Change this to test different queries
#TEST_UTTERANCE = "I'm looking for a spy movie with Pierce Brosnan from the 1990s"
TEST_UTTERANCE = "What Bond film did Daniel Craig star in that won awards?"
#TEST_UTTERANCE = "Recommend me a film"
#TEST_UTTERANCE = "I'm thinking of a film. Try to guess it."
#TEST_UTTERANCE = "I want a recommendation for action movies similar to Heat"

# %% [markdown]
# Setup & Imports - Database connection, LLM client, basic utilities
import json
import numpy as np
import requests
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "Gemma3:12b"

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

def extract_entities_llm(utterance: str) -> Dict[str, Any]:
    """Extract entities and terms using LLM with JSON schema validation"""
    
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
        
        return {
            'canonical_terms': canonical_terms,
            'entities': entities,
            'numbers': numbers,
            'dates': dates
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
    MATCH (e:Entity:Demo)
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
                MATCH (e:Entity:Demo)
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
            MATCH path = (start:Entity:Demo {{id: $anchor_id}})-[*1..{hops}]-(connected)
            WHERE connected:Entity:Demo OR connected:SlotValue
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
            MATCH (e:Entity:Demo)
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

# Execute retrieval
print("Starting graph retrieval...")

# Find anchor nodes using semantic similarity
anchors = find_anchor_nodes(u_sem)
print(f"Found {len(anchors)} anchor nodes:")
for anchor in anchors[:3]:  # Show top 3
    print(f"  {anchor['name']} ({anchor['id']}) - score: {anchor['s_combined']:.3f}")

# Link extracted entities to graph
linked_entity_ids = link_entities_to_graph(extraction_result['entities'])
print(f"\nLinked entities: {linked_entity_ids}")

# Expand into candidate subgraphs
candidates = expand_subgraphs(anchors)
print(f"\nGenerated {len(candidates)} candidate subgraphs:")
for i, candidate in enumerate(candidates[:3]):  # Show top 3
    print(f"  Candidate {i}: anchor={candidate['anchor_name']}, size={candidate['subgraph_size']}, score={candidate['retrieval_score']:.3f}")

# Store retrieval context R
retrieval_context_R = {
    'anchors': anchors,
    'linked_entity_ids': linked_entity_ids,
    'candidates': candidates,
    'expansion_params': {'hops': DEFAULTS['hops'], 'k_anchors': DEFAULTS['K_anchors']},
    'utterance': TEST_UTTERANCE
}

print(f"\nRetrieval context R created with {len(candidates)} candidates")

# %% [markdown]
# # Feature Generation - Create observed and predicted features for likelihood
# Generate u_struct_obs and expected features u' for each candidate

# %%
def compute_u_struct_obs(candidate: Dict[str, Any]) -> Dict[str, int]:
    """Compute observed structural features for a candidate subgraph"""
    
    try:
        # Query structural features of the subgraph
        cypher_query = """
        MATCH (anchor:Entity:Demo {id: $anchor_id})
        OPTIONAL MATCH (anchor)-[r]-(connected)
        WHERE connected.id IN $connected_ids
        RETURN 
            count(DISTINCT connected) as node_count,
            count(DISTINCT type(r)) as edge_type_count,
            collect(DISTINCT labels(connected)) as node_label_groups,
            collect(DISTINCT type(r)) as edge_types
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher_query, 
                               anchor_id=candidate['anchor_id'],
                               connected_ids=candidate['connected_ids'])
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
                    if label != 'Demo':  # Skip the Demo label
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
    """Generate expected terms that should appear if this candidate is correct"""
    
    expected_terms = []
    
    # Add node names (lowercased and split)
    for name in candidate.get('connected_names', []):
        if name:
            # Split compound names and add individual words
            words = name.lower().replace('-', ' ').replace(':', ' ').split()
            expected_terms.extend([w for w in words if len(w) > 2])
    
    # Add anchor name
    if candidate.get('anchor_name'):
        anchor_words = candidate['anchor_name'].lower().replace('-', ' ').split()
        expected_terms.extend([w for w in anchor_words if len(w) > 2])
    
    # Query for slot values (genres, etc.) associated with this subgraph
    try:
        cypher_query = """
        MATCH (anchor:Entity:Demo {id: $anchor_id})-[:HAS_SLOT]->(sv:SlotValue)
        RETURN sv.slot as slot, sv.value as value
        """
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher_query, anchor_id=candidate['anchor_id'])
            
            for record in result:
                if record['value']:
                    # Add slot values as expected terms
                    slot_words = record['value'].lower().split()
                    expected_terms.extend([w for w in slot_words if len(w) > 2])
                    
                    # Add slot type as term
                    if record['slot']:
                        expected_terms.append(record['slot'].lower())
    
    except Exception as e:
        logger.error(f"Failed to get slot values for {candidate['anchor_id']}: {e}")
    
    # Dedupe and limit
    expected_terms = list(dict.fromkeys(expected_terms))[:DEFAULTS['N_expected']]
    return expected_terms

def compute_delta_distances(observation_u: Dict, candidate: Dict, expected_terms: List[str]) -> Dict[str, float]:
    """Compute delta distances for semantic, structural, and terms channels"""
    
    # δ_sem: semantic distance (placeholder - would use actual subgraph embedding)
    # For now, use inverse of retrieval score as a proxy
    delta_sem = max(0.0, 1.0 - candidate.get('retrieval_score', 0.0))
    
    # δ_struct: structural distance (placeholder - needs expected structure from checklist)
    # For now, just use a default
    delta_struct = 0.3
    
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
def build_checklist_prior() -> Dict[str, float]:
    """Build prior over checklists based on domain frequency and context"""
    
    # Query available checklists
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("MATCH (c:Checklist:Demo) RETURN c.name as name, c.description as description")
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
    
    # Generic prior with slight bias toward identification-style tasks
    checklist_prior = {}
    total_mass = 0.95  # Leave 5% for "none-of-the-above"
    
    for checklist in checklists:
        name = checklist['name']
        if 'Identify' in name:
            checklist_prior[name] = 0.5  # Strong bias toward identification for this query type
        elif 'Recommend' in name:
            checklist_prior[name] = 0.3
        elif 'Verify' in name:
            checklist_prior[name] = 0.15
        else:
            checklist_prior[name] = 0.1
    
    # Normalize
    current_sum = sum(checklist_prior.values())
    if current_sum > 0:
        for name in checklist_prior:
            checklist_prior[name] = (checklist_prior[name] / current_sum) * total_mass
    
    checklist_prior['None'] = 0.05  # None-of-the-above option
    
    return checklist_prior

def build_goal_prior(checklist_prior: Dict[str, float]) -> Dict[str, float]:
    """Build prior over goals conditioned on checklist"""
    
    # Standard dialogue goals
    base_goals = {
        'identify': 0.4,   # "What movie is this?"
        'recommend': 0.2,  # "Suggest similar movies"
        'verify': 0.15,    # "Is this fact correct?"
        'explore': 0.15,   # "Tell me more about..."
        'act': 0.1         # "Do something with this info"
    }
    
    # Adjust based on active checklist (use top checklist for simplicity)
    top_checklist = max(checklist_prior.items(), key=lambda x: x[1])[0]
    
    if 'Identify' in top_checklist:
        base_goals['identify'] *= 1.5
        base_goals['verify'] *= 0.8
    elif 'Recommend' in top_checklist:
        base_goals['recommend'] *= 1.8
        base_goals['identify'] *= 0.7
    elif 'Verify' in top_checklist:
        base_goals['verify'] *= 1.6
        base_goals['explore'] *= 1.2
    
    # Normalize
    total = sum(base_goals.values())
    return {k: v/total for k, v in base_goals.items()}

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

def build_dialogue_act_prior(utterance: str) -> Dict[str, float]:
    """Build prior over dialogue acts based on utterance features"""
    
    utterance_lower = utterance.lower()
    
    # Simple heuristics based on utterance patterns
    dialogue_prior = {
        'clarify': 0.2,   # "What/which/how" questions
        'confirm': 0.2,   # "Is this..." statements  
        'request': 0.3,   # "I want/looking for" statements
        'provide': 0.3    # Declarative information
    }
    
    # Adjust based on patterns
    if any(word in utterance_lower for word in ['what', 'which', 'who', 'when', 'where', 'how']):
        dialogue_prior['clarify'] *= 1.5
        dialogue_prior['request'] *= 1.2
    
    if any(word in utterance_lower for word in ['looking for', 'want', 'need', 'find']):
        dialogue_prior['request'] *= 1.8
        dialogue_prior['provide'] *= 0.7
    
    if any(word in utterance_lower for word in ['is this', 'is that', 'correct', 'true']):
        dialogue_prior['confirm'] *= 1.6
        dialogue_prior['clarify'] *= 1.2
    
    if '?' in utterance:
        dialogue_prior['clarify'] *= 1.3
        dialogue_prior['request'] *= 1.2
    
    # Normalize
    total = sum(dialogue_prior.values())
    return {k: v/total for k, v in dialogue_prior.items()}

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

# Build all priors
print("Building priors over hidden states...")

# Checklist prior
checklist_prior = build_checklist_prior()
print(f"Checklist prior: {checklist_prior}")

# Goal prior
goal_prior = build_goal_prior(checklist_prior)
print(f"Goal prior: {goal_prior}")

# Subgraph prior
subgraph_prior = build_subgraph_prior(retrieval_context_R['candidates'])
print(f"Subgraph prior (top 3): {dict(list(subgraph_prior.items())[:3])}")

# Dialogue act prior
dialogue_act_prior = build_dialogue_act_prior(TEST_UTTERANCE)
print(f"Dialogue act prior: {dialogue_act_prior}")

# Novelty prior
novelty_prior = build_novelty_prior(retrieval_context_R['anchors'], observation_u)
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
            MATCH (ss:SlotSpec:Demo {checklist_name: $cl}) WHERE ss.required = true
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
    """Update goal posterior based on dialogue act and evidence"""
    
    goal_posterior = priors['goal'].copy()
    
    # Adjust based on dialogue act
    top_dialogue_act = max(dialogue_act_posterior.items(), key=lambda x: x[1])[0]
    
    if top_dialogue_act == 'request':
        goal_posterior['identify'] *= 1.3
        goal_posterior['recommend'] *= 1.1
    elif top_dialogue_act == 'clarify':
        goal_posterior['explore'] *= 1.4
        goal_posterior['verify'] *= 1.2
    elif top_dialogue_act == 'confirm':
        goal_posterior['verify'] *= 1.5
        goal_posterior['identify'] *= 1.1
    
    # Normalize
    total = sum(goal_posterior.values())
    if total > 0:
        goal_posterior = {k: v/total for k, v in goal_posterior.items()}
    
    return goal_posterior

# Perform posterior updates
print("Updating posterior beliefs...")

# Update subgraph posterior (main inference)
posterior_subgraph = update_posterior_subgraph(retrieval_context_R['candidates'], priors, observation_u)

# Get top candidate for conditioning other posteriors
top_subgraph_id = max(posterior_subgraph.items(), key=lambda x: x[1])[0]
top_candidate = next(c for c in retrieval_context_R['candidates'] if c['id'] == top_subgraph_id)

print(f"Top subgraph: {top_candidate['anchor_name']} (probability: {posterior_subgraph[top_subgraph_id]:.3f})")

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
            MATCH (ss:SlotSpec:Demo {checklist_name: $cl})
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

def calculate_eig_ask(slot_analysis: Dict) -> Dict[str, float]:
    """Calculate Expected Information Gain for asking about each slot"""
    
    eig_ask = {}
    
    for slot, info in slot_analysis.items():
        # Simple EIG approximation: higher entropy = higher potential gain
        base_eig = info['entropy']
        
        # Adjust for current confidence (lower confidence = higher gain potential)
        confidence_factor = 1.0 - info['confidence']
        
        # Adjust for evidence quality
        evidence_factor = {
            'missing': 1.2,
            'weak': 1.0,
            'structural': 0.8,
            'strong': 0.3
        }.get(info['evidence'], 1.0)
        
        # Combined EIG estimate
        eig = base_eig * confidence_factor * evidence_factor
        eig_ask[slot] = eig
    
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

def simulate_user_responses(target_slot: str, posteriors: Dict) -> List[Tuple[str, float]]:
    """Simulate possible user responses to asking about a slot"""
    
    responses = []
    
    # Generic response profiles by slot "kind" inferred from name
    slot_lower = target_slot.lower()
    if any(k in slot_lower for k in ['year', 'date', 'time']):
        responses = [("specific_value", 0.5), ("range", 0.3), ("approximate", 0.15), ("unknown", 0.05)]
    elif any(k in slot_lower for k in ['id', 'name', 'title']):
        responses = [("specific_value", 0.7), ("additional_clues", 0.2), ("unknown", 0.1)]
    else:
        responses = [("informative", 0.6), ("partial", 0.3), ("unknown", 0.1)]
    
    return responses

def estimate_entropy_reduction(response_type: str, target_slot: str, posteriors: Dict) -> float:
    """Estimate how much entropy would be reduced by a given response"""
    
    current_entropy = -sum(p * np.log2(p) for p in posteriors['subgraph'].values() if p > 1e-10)
    
    # Estimate entropy reduction based on response informativeness
    if response_type in ("specific_value", "confirm_current"):
        # Very informative - should collapse to one candidate
        reduction = current_entropy * 0.8
    elif response_type in ("range", "specific_year", "informative"):
        # Moderately informative - eliminates some candidates
        reduction = current_entropy * 0.5
    elif response_type in ("decade", "approximate", "partial"):
        # Somewhat informative - narrows down options
        reduction = current_entropy * 0.3
    elif response_type == "provide_different":
        # Changes the game - new candidates but resolves uncertainty
        reduction = current_entropy * 0.6
    else:  # unknown, approximate
        # Low information - small reduction
        reduction = current_entropy * 0.1
    
    return reduction

def estimate_followup_eig(response_type: str, target_slot: str, posteriors: Dict) -> float:
    """Estimate EIG of best follow-up action after user response"""
    
    # After user response, what would be the best next action's EIG?
    if response_type in ["specific_value", "confirm_current"]:
        # User gave definitive answer - next action likely ANSWER with high confidence
        return 0.1  # Low EIG because we're nearly done
    elif response_type in ["specific_year", "informative", "range"]:
        # Good info but may need one more clarification
        return 0.4  # Moderate EIG for final disambiguation
    elif response_type == "provide_different":
        # New actor mentioned - need to search/ask about their films
        return 0.7  # High EIG for exploring new branch
    elif response_type in ["decade", "partial"]:
        # Partial info - likely need another targeted question
        return 0.5  # Moderate EIG for follow-up question
    else:  # unknown
        # User doesn't know - may need to switch to SEARCH
        return 0.6  # Search becomes more valuable

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

def make_decision(posteriors: Dict, entropies: Dict, top_candidate: Dict, margin: float) -> Dict[str, Any]:
    """Decide between ANSWER, ASK, or SEARCH based on uncertainty and lookahead EIG"""
    
    # Decision thresholds
    ANSWER_CONFIDENCE_THRESHOLD = 0.70
    ANSWER_MARGIN_THRESHOLD = 0.20
    MAX_ENTROPY_THRESHOLD = 0.5
    
    top_prob = max(posteriors['subgraph'].values())
    
    # Get active checklist
    top_checklist = max(posteriors['checklist'].items(), key=lambda x: x[1])[0]
    
    # Analyze slots
    slot_analysis = analyze_slot_uncertainty(top_candidate, top_checklist)
    eig_ask_immediate = calculate_eig_ask(slot_analysis)
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
    
    # Check if we should ANSWER
    all_slot_entropies = [info['entropy'] for info in slot_analysis.values()]
    max_slot_entropy = max(all_slot_entropies) if all_slot_entropies else 0
    
    if (top_prob >= ANSWER_CONFIDENCE_THRESHOLD and 
        margin >= ANSWER_MARGIN_THRESHOLD and 
        max_slot_entropy < MAX_ENTROPY_THRESHOLD):
        
        decision['action'] = 'ANSWER'
        decision['target'] = top_candidate['anchor_name']
        decision['reasoning'] = f"High confidence ({top_prob:.3f}) and low uncertainty"
    
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
            decision['reasoning'] = f"Asking about '{best_ask_slot}' has highest lookahead EIG ({best_ask_eig:.3f}, immediate: {immediate_eig:.3f})"
        else:
            decision['action'] = 'SEARCH'
            decision['target'] = 'missing_facts'
            decision['reasoning'] = f"Search has higher EIG ({eig_search:.3f}) than asking"
    
    return decision

# Perform uncertainty analysis and decision making
print("Analyzing uncertainty and making decision...")

# Calculate entropies
entropies = calculate_entropies(posteriors)
print(f"Entropies: {entropies}")

# Make decision
decision = make_decision(posteriors, entropies, top_candidate, margin)

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
    top_candidate = system_state['top_candidate']
    
    # Build context for question generation (domain-agnostic)
    context = {
        "checklist": system_state.get('active_checklist', max(system_state.get('priors', {}).get('checklist', {"None":1.0}).items(), key=lambda x: x[1])[0]),
        "slot": target_slot,
        "top_candidate": {
            "anchor": top_candidate['anchor_name'],
            "connected": top_candidate.get('connected_names', [])[:3]
        },
        "confidence": decision['confidence'],
        "reasoning": f"Need to disambiguate {target_slot} to improve identification"
    }
    
    # Create prompt for question generation
    prompt = f"""Generate a natural, conversational question to ask the user. Context:

- Task: {context['checklist']} 
- Need to ask about: {context['slot']}
- Top candidate: {context['top_candidate']['anchor']}
- Confidence: {context['confidence']:.1%}
- Why: {context['reasoning']}

Generate ONE short, natural question that would help reduce uncertainty for the target. Be conversational and helpful.

Question:"""
    
    try:
        response = call_ollama_json(f"{prompt}\n\nReturn JSON with format: {{\"question\": \"your question here\"}}")
        if 'question' in response:
            return response['question']
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
    
    # Fallback to template-based question (generic)
    return f"Could you share the {target_slot} you're thinking of?"

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
    print(f"Top subgraph: {top_candidate['anchor_name']} (probability: {posteriors['subgraph'][top_subgraph_id]:.3f})")
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
print(f"System ready for next interaction!")

# %%
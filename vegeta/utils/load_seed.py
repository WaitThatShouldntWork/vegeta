#!/usr/bin/env python3
"""
VEGETA Database Loader - Complete data engineering pipeline for Neo4j

This script:
1. Executes seed.cypher to create the base graph structure
2. Generates embeddings for all entities using Ollama
3. Creates vector indexes for efficient similarity search
4. Sets up fulltext indexes for entity lookup
5. Validates data integrity and reports statistics

Usage:
    python load_seed.py [--reset] [--deep-reset] [--skip-embeddings] [--skip-indexes]
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError

# Setup logging with proper encoding for Windows
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('load_seed.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_DIMENSION = 768  # nomic-embed-text dimension
SEED_FILE = "study/seed.cypher"

class DatabaseLoader:
    def __init__(self):
        self.driver = None
        self.stats = {
            'entities_processed': 0,
            'embeddings_generated': 0,
            'indexes_created': 0,
            'errors': []
        }
    
    def connect(self) -> bool:
        """Establish Neo4j connection and test it"""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            
            # Test connection
            with self.driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("RETURN 'Connection successful' AS status")
                record = result.single()
                logger.info(f"✓ Neo4j connected: {record['status']}")
                return True
                
        except ServiceUnavailable as e:
            logger.error(f"✗ Neo4j connection failed: {e}")
            return False
    
    def test_ollama(self) -> bool:
        """Test Ollama connection and embedding model availability"""
        try:
            # Test connection
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error(f"✗ Ollama not accessible: HTTP {response.status_code}")
                return False
            
            # Check if embedding model is available
            models = [model['name'] for model in response.json().get('models', [])]
            if EMBEDDING_MODEL not in models:
                logger.warning(f"⚠ Model {EMBEDDING_MODEL} not found. Available: {models}")
                logger.info("Attempting to pull the model...")
                pull_response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/pull",
                    json={"name": EMBEDDING_MODEL},
                    timeout=300
                )
                if pull_response.status_code != 200:
                    logger.error(f"✗ Failed to pull {EMBEDDING_MODEL}")
                    return False
                logger.info(f"✓ Successfully pulled {EMBEDDING_MODEL}")
            else:
                logger.info(f"✓ Ollama model {EMBEDDING_MODEL} available")
            
            # Test embedding generation
            test_embedding = self.get_embedding("test")
            if test_embedding is not None:
                logger.info(f"✓ Embedding test successful (dim: {len(test_embedding)})")
                return True
            else:
                logger.error("✗ Embedding test failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Ollama test failed: {e}")
            return False
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using Ollama"""
        try:
            payload = {
                "model": EMBEDDING_MODEL,
                "prompt": text
            }
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings", 
                json=payload, 
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            embedding = np.array(result['embedding'])
            
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed for '{text}': {e}")
            return None
    
    def execute_cypher_file(self, file_path: str) -> bool:
        """Execute Cypher file with proper statement separation"""
        try:
            if not Path(file_path).exists():
                logger.error(f"✗ Seed file not found: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                cypher_content = f.read()

            logger.info(f"📁 Executing Cypher file: {file_path}")

            # Improved Cypher statement parsing
            import re

            # Remove comments (but preserve semicolons in comments)
            # Only remove comments that start after whitespace or at line start, not inside strings
            cleaned_content = re.sub(r'^\s*//.*$', '', cypher_content, flags=re.MULTILINE)

            statements = []
            current_statement = ""
            in_string = False
            string_char = None
            brace_depth = 0
            bracket_depth = 0

            i = 0
            while i < len(cleaned_content):
                char = cleaned_content[i]

                # Handle string literals
                if char in ['"', "'"] and (i == 0 or cleaned_content[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None

                # Track braces and brackets
                elif not in_string:
                    if char == '{':
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1
                    elif char == '[':
                        bracket_depth += 1
                    elif char == ']':
                        bracket_depth -= 1

                # Build current statement
                current_statement += char

                # Check for statement end - semicolon at top level (not in strings/braces/brackets)
                if (char == ';' and
                    not in_string and
                    brace_depth == 0 and
                    bracket_depth == 0):

                    statement = current_statement.strip()
                    if statement and not statement.startswith('//'):
                        statements.append(statement)
                    current_statement = ""

                i += 1

            # Add any remaining statement
            if current_statement.strip():
                statement = current_statement.strip()
                if statement and not statement.startswith('//'):
                    statements.append(statement)

            logger.info(f"Found {len(statements)} Cypher statements to execute")
            
            with self.driver.session(database=NEO4J_DATABASE) as session:
                for i, statement in enumerate(statements):
                    if statement:
                        try:
                            result = session.run(statement)
                            # Consume result to ensure execution
                            summary = result.consume()
                            if summary.counters:
                                logger.debug(f"Statement {i+1}: {summary.counters}")
                        except ClientError as e:
                            # Some statements might fail (like constraints that already exist)
                            logger.warning(f"Statement {i+1} warning: {e}")
                            # Continue with other statements
            
            logger.info("✓ Cypher file executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to execute Cypher file: {e}")
            self.stats['errors'].append(f"Cypher execution: {e}")
            return False
    
    def clear_database(self) -> bool:
        """Clear all nodes, relationships, indexes, and constraints"""
        try:
            logger.info("🗑️  Clearing existing data, indexes, and constraints...")
            with self.driver.session(database=NEO4J_DATABASE) as session:
                
                # 1. Drop all indexes first (vector indexes, fulltext, standard)
                self._drop_all_indexes(session)
                
                # 2. Drop all constraints
                self._drop_all_constraints(session)
                
                # 3. Clear all nodes and relationships
                result = session.run("MATCH (n) DETACH DELETE n")
                summary = result.consume()
                
                # 4. Clear orphaned SlotValues (if any)
                result = session.run("""
                    MATCH (sv:SlotValue) 
                    WHERE NOT (sv)<-[:HAS_SLOT]-() 
                    DELETE sv
                """)
                summary2 = result.consume()
                
                deleted = summary.counters.nodes_deleted + summary2.counters.nodes_deleted
                logger.info(f"✓ Cleared {deleted} nodes")
                return True
                
        except Exception as e:
            logger.error(f"✗ Failed to clear database: {e}")
            return False
    
    def clear_data_only(self) -> bool:
        """Clear only nodes and relationships (keep indexes and constraints)"""
        try:
            logger.info("🗑️  Clearing existing data (keeping indexes and constraints)...")
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Clear all nodes
                result = session.run("MATCH (n) DETACH DELETE n")
                summary = result.consume()
                
                # Clear orphaned SlotValues (if any)
                result = session.run("""
                    MATCH (sv:SlotValue) 
                    WHERE NOT (sv)<-[:HAS_SLOT]-() 
                    DELETE sv
                """)
                summary2 = result.consume()
                
                deleted = summary.counters.nodes_deleted + summary2.counters.nodes_deleted
                logger.info(f"✓ Cleared {deleted} nodes")
                return True
                
        except Exception as e:
            logger.error(f"✗ Failed to clear data: {e}")
            return False
    
    def _drop_all_indexes(self, session) -> None:
        """Drop all indexes (vector, fulltext, standard)"""
        try:
            logger.info("🗂️  Dropping all indexes...")
            
            # Get all indexes
            result = session.run("SHOW INDEXES")
            indexes = [record['name'] for record in result if record['name']]
            
            dropped_count = 0
            for index_name in indexes:
                try:
                    session.run(f"DROP INDEX {index_name}")
                    dropped_count += 1
                    logger.debug(f"  Dropped index: {index_name}")
                except ClientError as e:
                    # Some indexes might fail to drop (e.g., system indexes)
                    logger.debug(f"  Could not drop index {index_name}: {e}")
            
            logger.info(f"✓ Dropped {dropped_count} indexes")
            
        except Exception as e:
            logger.warning(f"⚠ Could not drop all indexes: {e}")
    
    def _drop_all_constraints(self, session) -> None:
        """Drop all constraints"""
        try:
            logger.info("🔒 Dropping all constraints...")
            
            # Get all constraints
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record['name'] for record in result if record['name']]
            
            dropped_count = 0
            for constraint_name in constraints:
                try:
                    session.run(f"DROP CONSTRAINT {constraint_name}")
                    dropped_count += 1
                    logger.debug(f"  Dropped constraint: {constraint_name}")
                except ClientError as e:
                    # Some constraints might fail to drop (e.g., system constraints)
                    logger.debug(f"  Could not drop constraint {constraint_name}: {e}")
            
            logger.info(f"✓ Dropped {dropped_count} constraints")
            
        except Exception as e:
            logger.warning(f"⚠ Could not drop all constraints: {e}")
    
    def generate_embeddings_only(self) -> bool:
        """Generate embeddings for all entities (without reloading data)"""
        try:
            logger.info("🧠 Generating embeddings for entities...")
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            with driver.session(database=NEO4J_DATABASE) as session:
                # Get all entities that need embeddings (Entity, Checklist, SlotSpec)
                result = session.run("""
                    MATCH (e)
                    WHERE e:Entity OR e:Checklist OR e:SlotSpec
                    RETURN CASE
                             WHEN e:Checklist THEN e.id
                             WHEN e:SlotSpec THEN e.id
                             ELSE e.id
                           END as id,
                           CASE
                             WHEN e:Checklist THEN e.name
                             WHEN e:SlotSpec THEN e.name
                             ELSE e.name
                           END as name,
                           CASE
                             WHEN e:Checklist THEN e.description
                             WHEN e:SlotSpec THEN e.name + ' for ' + e.checklist_name
                             ELSE coalesce(e.name, '')
                           END as description,
                           labels(e) as labels
                """)
                entities = list(result)

                logger.info(f"Found {len(entities)} entities to process")

                for i, entity in enumerate(entities):
                    entity_id = entity['id']
                    name = entity['name']
                    description = entity['description'] or ''
                    labels = entity['labels']

                    # Debug logging for Checklist and SlotSpec nodes
                    if 'Checklist' in labels or 'SlotSpec' in labels:
                        logger.info(f"🔍 Processing {labels}: {name} (desc: '{description}')")

                    # Create text for embedding (name + description)
                    # Handle None values properly and ensure we have some text to embed
                    text_parts = []
                    if name:
                        text_parts.append(name)
                    if description and description != name:
                        text_parts.append(description)

                    # If we have no text at all, use the ID as fallback
                    if not text_parts:
                        text_parts = [entity_id.split(':')[-1]]  # Use the last part of the ID

                    embed_text = ' '.join(text_parts)

                    # Debug logging for problematic nodes
                    if not embed_text.strip():
                        logger.warning(f"⚠ Empty embed_text for node {entity_id} (labels: {labels}, name: '{name}', desc: '{description}')")

                    # Generate semantic embedding
                    sem_emb = self.get_embedding(embed_text)
                    if sem_emb is not None:
                        # Debug logging for Checklist and SlotSpec
                        if 'Checklist' in labels or 'SlotSpec' in labels:
                            logger.info(f"💾 Storing embedding for {labels}: {name} (size: {len(sem_emb)})")

                        # Store embedding in database - handle different node types
                        try:
                            result = session.run("""
                                MATCH (e {id: $id})
                                SET e.sem_emb = $embedding,
                                    e.embed_text = $embed_text,
                                    e.embed_timestamp = datetime()
                                RETURN e.id as stored_id, labels(e) as stored_labels
                            """, id=entity_id, embedding=sem_emb.tolist(), embed_text=embed_text)

                            # Verify storage
                            stored = result.single()
                            if stored:
                                stored_id = stored['stored_id']
                                stored_labels = stored['stored_labels']
                                if 'Checklist' in labels or 'SlotSpec' in labels:
                                    logger.info(f"✅ Verified storage for {stored_labels}: {name} (stored_id: {stored_id})")
                            else:
                                logger.error(f"❌ Storage failed for {labels}: {name} - no result returned")
                        except Exception as store_error:
                            logger.error(f"❌ Storage error for {labels}: {name} - {store_error}")

                        self.stats['embeddings_generated'] += 1

                        if (i + 1) % 10 == 0:
                            logger.info(f"  Processed {i + 1}/{len(entities)} entities")
                    else:
                        logger.warning(f"Failed to generate embedding for entity: {entity_id}")

            logger.info(f"✓ Generated embeddings for {len(entities)} entities")
            return True

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return False

    def generate_embeddings(self) -> bool:
        """Generate embeddings for all entities"""
        try:
            logger.info("🧠 Generating embeddings for entities...")
            
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Get all entities that need embeddings (Entity, Checklist, SlotSpec)
                result = session.run("""
                    MATCH (e)
                    WHERE e:Entity OR e:Checklist OR e:SlotSpec
                    RETURN CASE
                             WHEN e:Checklist THEN e.id
                             WHEN e:SlotSpec THEN e.id
                             ELSE e.id
                           END as id,
                           CASE
                             WHEN e:Checklist THEN e.name
                             WHEN e:SlotSpec THEN e.name
                             ELSE e.name
                           END as name,
                           CASE
                             WHEN e:Checklist THEN e.description
                             WHEN e:SlotSpec THEN e.name + ' for ' + e.checklist_name
                             ELSE coalesce(e.name, '')
                           END as description,
                           labels(e) as labels
                """)
                entities = list(result)
                
                logger.info(f"Found {len(entities)} entities to process")
                
                for i, entity in enumerate(entities):
                    entity_id = entity['id']
                    name = entity['name']
                    description = entity['description'] or ''
                    labels = entity['labels']

                    # Debug logging for Checklist and SlotSpec nodes
                    if 'Checklist' in labels or 'SlotSpec' in labels:
                        logger.info(f"🔍 Processing {labels}: {name} (desc: '{description}')")

                    # Create text for embedding (name + description)
                    # Handle None values properly and ensure we have some text to embed
                    text_parts = []
                    if name:
                        text_parts.append(name)
                    if description and description != name:
                        text_parts.append(description)

                    # If we have no text at all, use the ID as fallback
                    if not text_parts:
                        text_parts = [entity_id.split(':')[-1]]  # Use the last part of the ID

                    embed_text = ' '.join(text_parts)

                    # Debug logging for problematic nodes
                    if not embed_text.strip():
                        logger.warning(f"⚠ Empty embed_text for node {entity_id} (labels: {labels}, name: '{name}', desc: '{description}')")

                    # Generate semantic embedding
                    sem_emb = self.get_embedding(embed_text)
                    if sem_emb is not None:
                        # Debug logging for Checklist and SlotSpec
                        if 'Checklist' in labels or 'SlotSpec' in labels:
                            logger.info(f"💾 Storing embedding for {labels}: {name} (size: {len(sem_emb)})")

                        # Store embedding in database - handle different node types
                        try:
                            result = session.run("""
                                MATCH (e {id: $id})
                                SET e.sem_emb = $embedding,
                                    e.embed_text = $embed_text,
                                    e.embed_timestamp = datetime()
                                RETURN e.id as stored_id, labels(e) as stored_labels
                            """, id=entity_id, embedding=sem_emb.tolist(), embed_text=embed_text)

                            # Verify storage
                            stored = result.single()
                            if stored:
                                stored_id = stored['stored_id']
                                stored_labels = stored['stored_labels']
                                if 'Checklist' in labels or 'SlotSpec' in labels:
                                    logger.info(f"✅ Verified storage for {stored_labels}: {name} (stored_id: {stored_id})")
                            else:
                                logger.error(f"❌ Storage failed for {labels}: {name} - no result returned")
                        except Exception as store_error:
                            logger.error(f"❌ Storage error for {labels}: {name} - {store_error}")

                        self.stats['embeddings_generated'] += 1
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"  Processed {i + 1}/{len(entities)} entities")
                    else:
                        logger.warning(f"Failed to generate embedding for entity: {entity_id}")
                        self.stats['errors'].append(f"Embedding failed: {entity_id}")
                
                self.stats['entities_processed'] = len(entities)
                logger.info(f"✓ Generated embeddings for {self.stats['embeddings_generated']} entities")
                return True
                
        except Exception as e:
            logger.error(f"✗ Failed to generate embeddings: {e}")
            self.stats['errors'].append(f"Embedding generation: {e}")
            return False
    
    def create_vector_indexes(self) -> bool:
        """Create vector indexes for similarity search with robust version detection"""
        try:
            logger.info("📊 Creating vector indexes...")

            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Check Neo4j version and capabilities
                try:
                    # Get server info to determine capabilities
                    result = session.run("CALL dbms.components() YIELD name, versions, edition")
                    server_info = list(result)
                    neo4j_version = None
                    for component in server_info:
                        if component['name'] == 'Neo4j Kernel':
                            neo4j_version = component['versions'][0]
                            break

                    logger.info(f"Detected Neo4j version: {neo4j_version}")

                    # Try to create vector index for semantic embeddings
                    try:
                        if neo4j_version and neo4j_version.startswith('5'):
                            # Neo4j 5.x vector index syntax
                            session.run(f"""
                                CREATE VECTOR INDEX entity_semantic_index IF NOT EXISTS
                                FOR (e:Entity) ON (e.sem_emb)
                                OPTIONS {{
                                    indexConfig: {{
                                        `vector.dimensions`: {EMBEDDING_DIMENSION},
                                        `vector.similarity_function`: 'cosine'
                                    }}
                                }}
                            """)
                            logger.info("✓ Created Neo4j 5.x semantic embedding vector index")
                        else:
                            # Try alternative syntax or fallback
                            try:
                                session.run(f"""
                                    CREATE VECTOR INDEX entity_semantic_index IF NOT EXISTS
                                    FOR (e:Entity) ON (e.sem_emb)
                                    OPTIONS {{
                                        indexConfig: {{
                                            `vector.dimensions`: {EMBEDDING_DIMENSION},
                                            `vector.similarity_function`: 'cosine'
                                        }}
                                    }}
                                """)
                                logger.info("✓ Created semantic embedding vector index")
                            except ClientError as ve:
                                if "vector" in str(ve).lower() or "not supported" in str(ve).lower():
                                    logger.warning("⚠ Vector indexes not supported - embeddings will be stored but not indexed")
                                    logger.info("Consider upgrading to Neo4j 5.0+ for vector index support")
                                    # Still count as successful since embeddings are stored
                                else:
                                    raise ve

                        self.stats['indexes_created'] += 1

                    except ClientError as e:
                        if "already exists" in str(e):
                            logger.info("📋 Semantic embedding vector index already exists")
                            self.stats['indexes_created'] += 1
                        elif "vector" in str(e).lower():
                            logger.warning("⚠ Vector indexes not supported in this Neo4j version")
                            logger.info("Embeddings will be stored but similarity search will be slower")
                        else:
                            logger.warning(f"Vector index creation issue: {e}")
                            # Continue - embeddings are still valuable even without vector index

                except Exception as e:
                    logger.warning(f"Could not determine Neo4j version: {e}")
                    logger.info("Proceeding with basic vector index attempt...")

                # Additional vector indexes for different entity types
                try:
                    # Index for Award entities if they have embeddings
                    session.run("""
                        CREATE VECTOR INDEX award_semantic_index IF NOT EXISTS
                        FOR (a:Entity:Award) ON (a.sem_emb)
                        OPTIONS {
                            indexConfig: {
                                `vector.dimensions`: $dimensions,
                                `vector.similarity_function`: 'cosine'
                            }
                        }
                    """, dimensions=EMBEDDING_DIMENSION)
                    logger.info("✓ Created award semantic vector index")
                    self.stats['indexes_created'] += 1
                except ClientError as e:
                    if "already exists" not in str(e):
                        logger.debug(f"Award vector index creation skipped: {e}")

                try:
                    # Index for Person entities
                    session.run("""
                        CREATE VECTOR INDEX person_semantic_index IF NOT EXISTS
                        FOR (p:Entity:Person) ON (p.sem_emb)
                        OPTIONS {
                            indexConfig: {
                                `vector.dimensions`: $dimensions,
                                `vector.similarity_function`: 'cosine'
                            }
                        }
                    """, dimensions=EMBEDDING_DIMENSION)
                    logger.info("✓ Created person semantic vector index")
                    self.stats['indexes_created'] += 1
                except ClientError as e:
                    if "already exists" not in str(e):
                        logger.debug(f"Person vector index creation skipped: {e}")

                return True

        except Exception as e:
            logger.error(f"✗ Failed to create vector indexes: {e}")
            self.stats['errors'].append(f"Vector index creation: {e}")
            return False
    
    def create_fulltext_indexes(self) -> bool:
        """Create fulltext indexes for entity lookup"""
        try:
            logger.info("🔍 Creating fulltext indexes...")

            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Entity names and aliases index
                try:
                    session.run("""
                        CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
                        FOR (e:Entity) ON EACH [e.name, e.aliases]
                    """)
                    logger.info("✓ Created entity name fulltext index")
                    self.stats['indexes_created'] += 1
                except ClientError as e:
                    if "already exists" in str(e):
                        logger.info("📋 Entity fulltext index already exists")
                    else:
                        logger.warning(f"Entity fulltext index creation issue: {e}")

                # Film-specific index for plot and title search
                try:
                    session.run("""
                        CREATE FULLTEXT INDEX film_content_fulltext IF NOT EXISTS
                        FOR (f:Film) ON EACH [f.name]
                    """)
                    logger.info("✓ Created film content fulltext index")
                    self.stats['indexes_created'] += 1
                except ClientError as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Film fulltext index creation issue: {e}")

                # Award names index
                try:
                    session.run("""
                        CREATE FULLTEXT INDEX award_name_fulltext IF NOT EXISTS
                        FOR (a:Award) ON EACH [a.name]
                    """)
                    logger.info("✓ Created award name fulltext index")
                    self.stats['indexes_created'] += 1
                except ClientError as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Award fulltext index creation issue: {e}")

                # Person names index
                try:
                    session.run("""
                        CREATE FULLTEXT INDEX person_name_fulltext IF NOT EXISTS
                        FOR (p:Person) ON EACH [p.name, p.aliases]
                    """)
                    logger.info("✓ Created person name fulltext index")
                    self.stats['indexes_created'] += 1
                except ClientError as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Person fulltext index creation issue: {e}")

                return True

        except Exception as e:
            logger.error(f"✗ Failed to create fulltext indexes: {e}")
            self.stats['errors'].append(f"Fulltext index creation: {e}")
            return False
    
    def create_standard_indexes(self) -> bool:
        """Create standard property indexes for performance"""
        try:
            logger.info("⚡ Creating standard property indexes...")
            
            index_definitions = [
                # Core entity and type indexes
                "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
                "CREATE INDEX type_name_index IF NOT EXISTS FOR (t:Type) ON (t.name)",
                "CREATE INDEX relation_type_name_index IF NOT EXISTS FOR (r:RelationType) ON (r.name)",
                "CREATE INDEX slot_value_composite_index IF NOT EXISTS FOR (sv:SlotValue) ON (sv.slot, sv.value)",
                "CREATE INDEX document_url_index IF NOT EXISTS FOR (d:Document) ON (d.source_url)",

                # Fact and relationship indexes for better query performance
                "CREATE INDEX fact_kind_index IF NOT EXISTS FOR (f:Fact) ON (f.kind)",
                "CREATE INDEX fact_confidence_index IF NOT EXISTS FOR (f:Fact) ON (f.confidence)",
                "CREATE INDEX checklist_name_index IF NOT EXISTS FOR (c:Checklist) ON (c.name)",
                "CREATE INDEX slotspec_checklist_index IF NOT EXISTS FOR (ss:SlotSpec) ON (ss.checklist_name)",

                # Entity subtype indexes
                "CREATE INDEX film_entity_id_index IF NOT EXISTS FOR (f:Film) ON (f.id)",
                "CREATE INDEX person_entity_id_index IF NOT EXISTS FOR (p:Person) ON (p.id)",
                "CREATE INDEX award_entity_id_index IF NOT EXISTS FOR (a:Award) ON (a.id)",
                "CREATE INDEX year_entity_id_index IF NOT EXISTS FOR (y:Year) ON (y.id)",
                "CREATE INDEX year_value_index IF NOT EXISTS FOR (y:Year) ON (y.value)",

                # Section and paragraph indexes for provenance
                "CREATE INDEX section_doc_order_index IF NOT EXISTS FOR (s:Section) ON (s.doc_url, s.order)",
                "CREATE INDEX paragraph_doc_order_index IF NOT EXISTS FOR (p:Paragraph) ON (p.doc_url, p.order)",
                "CREATE INDEX sentence_doc_order_index IF NOT EXISTS FOR (snt:Sentence) ON (snt.doc_url, snt.order)"
            ]
            
            with self.driver.session(database=NEO4J_DATABASE) as session:
                for index_def in index_definitions:
                    try:
                        session.run(index_def)
                        self.stats['indexes_created'] += 1
                    except ClientError as e:
                        if "already exists" in str(e):
                            logger.debug(f"Index already exists: {index_def}")
                        else:
                            logger.warning(f"Index creation issue: {e}")
                
                logger.info(f"✓ Created/verified {len(index_definitions)} standard indexes")
                return True
                
        except Exception as e:
            logger.error(f"✗ Failed to create standard indexes: {e}")
            self.stats['errors'].append(f"Standard index creation: {e}")
            return False
    
    def validate_data(self) -> Dict[str, Any]:
        """Validate data integrity and gather statistics"""
        try:
            logger.info("🔍 Validating data integrity...")
            
            validation_results = {}
            
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Count various node types
                counts = {}
                node_types = ['Entity', 'Type', 'SlotValue', 'Document', 'Checklist', 'Fact',
                            'Section', 'Paragraph', 'Sentence', 'RelationType']

                for node_type in node_types:
                    result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                    counts[node_type] = result.single()['count']

                # Count entity subtypes
                subtype_counts = {}
                entity_subtypes = ['Entity:Film', 'Entity:Person', 'Entity:Award', 'Entity:Year',
                                 'Entity:Rating', 'Entity:Genre']

                for subtype in entity_subtypes:
                    result = session.run(f"MATCH (n:{subtype}) RETURN count(n) as count")
                    subtype_counts[subtype] = result.single()['count']

                validation_results['node_counts'] = counts
                validation_results['subtype_counts'] = subtype_counts

                # Check Fact types distribution
                result = session.run("""
                    MATCH (f:Fact)
                    RETURN f.kind as fact_kind, count(*) as count
                    ORDER BY count DESC
                """)
                fact_types = {record['fact_kind']: record['count'] for record in result}
                validation_results['fact_types'] = fact_types
                
                # Check embeddings
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN count(e) as total_entities,
                           count(e.sem_emb) as entities_with_embeddings
                """)
                embedding_stats = result.single()
                validation_results['embedding_coverage'] = {
                    'total_entities': embedding_stats['total_entities'],
                    'with_embeddings': embedding_stats['entities_with_embeddings'],
                    'coverage_pct': (embedding_stats['entities_with_embeddings'] / 
                                   max(embedding_stats['total_entities'], 1)) * 100
                }
                
                # Check constraints
                result = session.run("SHOW CONSTRAINTS")
                constraints = [record['name'] for record in result]
                validation_results['constraints'] = constraints
                
                # Check indexes
                result = session.run("SHOW INDEXES")
                indexes = [record['name'] for record in result]
                validation_results['indexes'] = indexes
                
                logger.info("✓ Data validation completed")
                return validation_results
                
        except Exception as e:
            logger.error(f"✗ Data validation failed: {e}")
            return {'error': str(e)}
    
    def print_statistics(self, validation_results: Dict[str, Any]):
        """Print comprehensive statistics"""
        print("\n" + "="*60)
        print("📊 DATABASE LOADING STATISTICS")
        print("="*60)
        
        # Processing stats
        print(f"Entities processed: {self.stats['entities_processed']}")
        print(f"Embeddings generated: {self.stats['embeddings_generated']}")
        print(f"Indexes created: {self.stats['indexes_created']}")
        print(f"Errors encountered: {len(self.stats['errors'])}")
        
        if validation_results and 'node_counts' in validation_results:
            print("\n📋 NODE COUNTS:")
            for node_type, count in validation_results['node_counts'].items():
                print(f"  {node_type}: {count}")

            if 'subtype_counts' in validation_results:
                print("\n🎭 ENTITY SUBTYPES:")
                for subtype, count in validation_results['subtype_counts'].items():
                    if count > 0:
                        print(f"  {subtype}: {count}")

            if 'fact_types' in validation_results:
                print("\n🔍 FACT TYPES:")
                for fact_type, count in validation_results['fact_types'].items():
                    print(f"  {fact_type}: {count}")

            print("\n🧠 EMBEDDING COVERAGE:")
            emb_stats = validation_results['embedding_coverage']
            print(f"  Total entities: {emb_stats['total_entities']}")
            print(f"  With embeddings: {emb_stats['with_embeddings']}")
            print(f"  Coverage: {emb_stats['coverage_pct']:.1f}%")

            print(f"\n📊 CONSTRAINTS: {len(validation_results.get('constraints', []))}")
            print(f"📊 INDEXES: {len(validation_results.get('indexes', []))}")
        
        if self.stats['errors']:
            print("\n⚠️  ERRORS:")
            for error in self.stats['errors']:
                print(f"  - {error}")
        
        print("="*60)
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("✓ Database connection closed")

def main():
    parser = argparse.ArgumentParser(description='Load VEGETA seed data with full data engineering pipeline')
    parser.add_argument('--reset', action='store_true', help='Clear existing nodes and relationships before loading')
    parser.add_argument('--deep-reset', action='store_true', help='Clear everything: nodes, relationships, indexes, and constraints')
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip embedding generation')
    parser.add_argument('--skip-indexes', action='store_true', help='Skip index creation')
    parser.add_argument('--seed-file', default=SEED_FILE, help='Path to seed.cypher file')
    
    args = parser.parse_args()
    
    loader = DatabaseLoader()
    
    try:
        # Connect to databases
        if not loader.connect():
            sys.exit(1)
        
        if not args.skip_embeddings and not loader.test_ollama():
            logger.error("Ollama test failed. Use --skip-embeddings to proceed without embeddings.")
            sys.exit(1)
        
        # Clear existing data if requested
        if args.deep_reset:
            if not loader.clear_database():
                sys.exit(1)
        elif args.reset:
            if not loader.clear_data_only():
                sys.exit(1)
        
        # Execute seed file
        if not loader.execute_cypher_file(args.seed_file):
            sys.exit(1)
        
        # Create standard indexes first
        if not args.skip_indexes:
            loader.create_standard_indexes()
            loader.create_fulltext_indexes()
        
        # Generate embeddings
        if not args.skip_embeddings:
            if not loader.generate_embeddings():
                logger.warning("Embedding generation failed, continuing...")
        
        # Create vector indexes (after embeddings)
        if not args.skip_indexes and not args.skip_embeddings:
            loader.create_vector_indexes()
        
        # Validate and report
        validation_results = loader.validate_data()
        loader.print_statistics(validation_results)
        
        logger.info("🎉 Database loading completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Loading interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        loader.close()

if __name__ == "__main__":
    main()

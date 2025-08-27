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
    python load_seed.py [--reset] [--skip-embeddings] [--skip-indexes]
"""

import os
import sys
import json
import time
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
            
            # Better statement parsing - split on ';' but only when it's at end of line
            # and not inside strings
            import re
            
            # Remove comments
            cleaned_content = re.sub(r'//.*$', '', cypher_content, flags=re.MULTILINE)
            
            # Split on semicolons that are followed by whitespace/newline (not in strings)
            # This is a simple heuristic - for production would use proper Cypher parser
            statements = []
            current_statement = ""
            
            for line in cleaned_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                current_statement += " " + line
                
                # Check if this line ends a statement (ends with semicolon)
                if line.endswith(';'):
                    statement = current_statement.strip()
                    if statement and not statement.startswith('//'):
                        statements.append(statement)
                    current_statement = ""
            
            # Add any remaining statement
            if current_statement.strip():
                statements.append(current_statement.strip())
            
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
        """Clear all Demo nodes and relationships"""
        try:
            logger.info("🗑️  Clearing existing Demo data...")
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Clear Demo nodes
                result = session.run("MATCH (n:Demo) DETACH DELETE n")
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
            logger.error(f"✗ Failed to clear database: {e}")
            return False
    
    def generate_embeddings(self) -> bool:
        """Generate embeddings for all entities"""
        try:
            logger.info("🧠 Generating embeddings for entities...")
            
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Get all entities that need embeddings
                result = session.run("""
                    MATCH (e:Entity:Demo)
                    RETURN e.id as id, e.name as name, e.aliases as aliases,
                           coalesce(e.plot, e.summary, '') as description
                """)
                entities = list(result)
                
                logger.info(f"Found {len(entities)} entities to process")
                
                for i, entity in enumerate(entities):
                    entity_id = entity['id']
                    name = entity['name']
                    aliases = entity['aliases'] or []
                    description = entity['description'] or ''
                    
                    # Create text for embedding (name + aliases + description)
                    # Handle None values properly
                    text_parts = [name] if name else []
                    if aliases:
                        text_parts.extend([alias for alias in aliases if alias])
                    if description:
                        text_parts.append(description)
                    embed_text = ' '.join(text_parts)
                    
                    # Generate semantic embedding
                    sem_emb = self.get_embedding(embed_text)
                    if sem_emb is not None:
                        # Store embedding in database
                        session.run("""
                            MATCH (e:Entity:Demo {id: $id})
                            SET e.sem_emb = $embedding,
                                e.embed_text = $embed_text,
                                e.embed_timestamp = datetime()
                        """, id=entity_id, embedding=sem_emb.tolist(), embed_text=embed_text)
                        
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
        """Create vector indexes for similarity search"""
        try:
            logger.info("📊 Creating vector indexes...")
            
            with self.driver.session(database=NEO4J_DATABASE) as session:
                # Check Neo4j version for vector index support
                try:
                    # Create vector index for semantic embeddings
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
                    self.stats['indexes_created'] += 1
                    
                except ClientError as e:
                    if "vector" in str(e).lower():
                        logger.warning("⚠ Vector indexes not supported in this Neo4j version")
                        logger.info("Consider upgrading to Neo4j 5.0+ for vector index support")
                    else:
                        logger.warning(f"Vector index creation issue: {e}")
                
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
                try:
                    # Create fulltext index for entity names and aliases
                    session.run("""
                        CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
                        FOR (e:Entity) ON EACH [e.name, e.aliases]
                    """)
                    logger.info("✓ Created entity name fulltext index")
                    self.stats['indexes_created'] += 1
                    
                except ClientError as e:
                    if "already exists" in str(e):
                        logger.info("📋 Fulltext index already exists")
                    else:
                        logger.warning(f"Fulltext index creation issue: {e}")
                
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
                "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
                "CREATE INDEX type_name_index IF NOT EXISTS FOR (t:Type) ON (t.name)",
                "CREATE INDEX relation_type_name_index IF NOT EXISTS FOR (r:RelationType) ON (r.name)",
                "CREATE INDEX slot_value_composite_index IF NOT EXISTS FOR (sv:SlotValue) ON (sv.slot, sv.value)",
                "CREATE INDEX document_url_index IF NOT EXISTS FOR (d:Document) ON (d.source_url)"
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
                node_types = ['Entity', 'Type', 'SlotValue', 'Document', 'Checklist', 'Fact']
                
                for node_type in node_types:
                    result = session.run(f"MATCH (n:{node_type}:Demo) RETURN count(n) as count")
                    counts[node_type] = result.single()['count']
                
                validation_results['node_counts'] = counts
                
                # Check embeddings
                result = session.run("""
                    MATCH (e:Entity:Demo)
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
    parser.add_argument('--reset', action='store_true', help='Clear existing Demo data before loading')
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
        if args.reset:
            if not loader.clear_database():
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

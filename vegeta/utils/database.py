"""
Database connection and management utilities
"""

import logging
from typing import Dict, Any, Optional, List
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError

from ..core.exceptions import ConnectionError, RetrievalError

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Neo4j database connection and query management
    """
    
    def __init__(self, db_config: Dict[str, str]):
        self.config = db_config
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.driver = GraphDatabase.driver(
                self.config['uri'],
                auth=(self.config['username'], self.config['password'])
            )
            logger.info("✓ Database connection established")
        except ServiceUnavailable as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.driver.session(database=self.config['database']) as session:
                result = session.run("RETURN 'Connection test' AS status")
                record = result.single()
                return record is not None
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        try:
            with self.driver.session(database=self.config['database']) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except ClientError as e:
            raise RetrievalError(f"Query execution failed: {e}")
        except Exception as e:
            raise ConnectionError(f"Database operation failed: {e}")
    
    def execute_query_single(self, query: str, parameters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Execute a query expecting a single result"""
        results = self.execute_query(query, parameters)
        return results[0] if results else None
    
    def execute_transaction(self, queries: List[tuple]) -> List[Any]:
        """Execute multiple queries in a transaction"""
        try:
            with self.driver.session(database=self.config['database']) as session:
                def tx_function(tx):
                    results = []
                    for query, params in queries:
                        result = tx.run(query, params or {})
                        results.append([record.data() for record in result])
                    return results
                
                return session.execute_write(tx_function)
        except Exception as e:
            raise RetrievalError(f"Transaction failed: {e}")
    
    def get_node_count(self, label: str) -> int:
        """Get count of nodes with specific label"""
        query = f"MATCH (n:{label}) RETURN count(n) as count"
        result = self.execute_query_single(query)
        return result['count'] if result else 0
    
    def get_entities_with_embeddings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get entities that have semantic embeddings"""
        query = """
        MATCH (e:Entity)
        WHERE e.sem_emb IS NOT NULL
        RETURN e.id as id, e.name as name, e.sem_emb as sem_emb, labels(e) as labels
        LIMIT $limit
        """
        return self.execute_query(query, {'limit': limit})
    
    def get_entities_by_fulltext(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities using fulltext index"""
        query = """
        CALL db.index.fulltext.queryNodes('entity_name_fulltext', $search_term)
        YIELD node, score
        RETURN node.id as id, node.name as name, labels(node) as labels, score
        LIMIT $limit
        """
        try:
            return self.execute_query(query, {'search_term': search_term, 'limit': limit})
        except Exception:
            # Fallback to simple text search if fulltext index doesn't exist
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($search_term)
               OR any(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($search_term))
            RETURN e.id as id, e.name as name, labels(e) as labels
            LIMIT $limit
            """
            return self.execute_query(query, {'search_term': search_term, 'limit': limit})
    
    def get_checklist_specs(self, checklist_name: str) -> List[Dict[str, Any]]:
        """Get SlotSpec requirements for a checklist"""
        query = """
        MATCH (ss:SlotSpec {checklist_name: $checklist_name})
        RETURN ss.name AS name, ss.expect_labels AS expect_labels, 
               ss.required AS required, ss.cardinality AS cardinality
        ORDER BY ss.required DESC, ss.name
        """
        return self.execute_query(query, {'checklist_name': checklist_name})
    
    def get_available_checklists(self) -> List[Dict[str, Any]]:
        """Get all available checklists"""
        query = """
        MATCH (c:Checklist)
        RETURN c.name as name, c.description as description
        """
        return self.execute_query(query)
    
    def get_entity_neighbors(self, entity_id: str, hops: int = 1) -> Dict[str, Any]:
        """Get neighbors of an entity within specified hops"""
        query = f"""
        MATCH (start:Entity {{id: $entity_id}})
        MATCH path = (start)-[*1..{hops}]-(connected)
        WHERE connected:Entity OR connected:SlotValue
        RETURN start.id as anchor_id,
               collect(DISTINCT connected.id) as connected_ids,
               collect(DISTINCT connected.name) as connected_names,
               collect(DISTINCT labels(connected)) as connected_labels,
               count(DISTINCT connected) as subgraph_size
        """
        result = self.execute_query_single(query, {'entity_id': entity_id})
        return result or {
            'anchor_id': entity_id,
            'connected_ids': [],
            'connected_names': [],
            'connected_labels': [],
            'subgraph_size': 0
        }
    
    def get_slot_values_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all slot values for an entity"""
        query = """
        MATCH (e:Entity {id: $entity_id})-[:HAS_SLOT]->(sv:SlotValue)
        RETURN sv.slot as slot, sv.value as value
        """
        return self.execute_query(query, {'entity_id': entity_id})
    
    def get_slot_values_for_entities_batch(self, entity_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Get all slot values for multiple entities in one query (BATCH VERSION)"""
        if not entity_ids:
            return {}
            
        query = """
        MATCH (e:Entity)-[:HAS_SLOT]->(sv:SlotValue)
        WHERE e.id IN $entity_ids
        RETURN e.id as entity_id, sv.slot as slot, sv.value as value
        """
        results = self.execute_query(query, {'entity_ids': entity_ids})
        
        # Group by entity_id
        slot_values_by_entity = {}
        for result in results:
            entity_id = result['entity_id']
            if entity_id not in slot_values_by_entity:
                slot_values_by_entity[entity_id] = []
            slot_values_by_entity[entity_id].append({
                'slot': result['slot'],
                'value': result['value']
            })
        
        # Ensure all requested entities have an entry (even if empty)
        for entity_id in entity_ids:
            if entity_id not in slot_values_by_entity:
                slot_values_by_entity[entity_id] = []
                
        return slot_values_by_entity
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("✓ Database connection closed")

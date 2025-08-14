from __future__ import annotations

from typing import Dict, Any

from db.neo4j_client import get_driver


def upsert_fact_with_source(*, subject_name: str, relation_name: str, object_name: str, source_url: str, confidence: float = 0.7) -> Dict[str, Any] | None:
	"""Create a reified Fact with SUBJECT/PREDICATE/OBJECT and attach HAS_SOURCE to a Document.

	Returns a small summary dict or None if Neo4j is not configured.
	"""
	driver = get_driver()
	if driver is None:
		return None
	query = """
	MERGE (sub:Entity {name:$sub})
	MERGE (obj:Entity {name:$obj})
	MERGE (rt:RelationType {name:$rel})
	MERGE (fa:Fact {kind:$rel, subject:$sub, object:$obj})
	  ON CREATE SET fa.confidence=$conf
	  ON MATCH SET fa.confidence = CASE WHEN coalesce(fa.confidence,0) < $conf THEN $conf ELSE fa.confidence END
	MERGE (fa)-[:SUBJECT]->(sub)
	MERGE (fa)-[:PREDICATE]->(rt)
	MERGE (fa)-[:OBJECT]->(obj)
	MERGE (d:Document {source_url:$url})
	MERGE (fa)-[:HAS_SOURCE {support:'web'}]->(d)
    RETURN elementId(fa) AS id
	"""
	with driver.session() as session:
		row = session.run(query, sub=subject_name, obj=object_name, rel=relation_name, conf=float(confidence), url=source_url).single()
		return {"fact_id": row["id"], "subject": subject_name, "relation": relation_name, "object": object_name, "url": source_url}


def fact_exists(*, subject_name: str, relation_name: str, object_name: str) -> bool:
	"""Return True if a reified Fact with (kind, subject, object) already exists."""
	driver = get_driver()
	if driver is None:
		return False
	q = """
	MATCH (fa:Fact {kind:$rel, subject:$sub, object:$obj}) RETURN count(fa) AS c
	"""
	with driver.session() as session:
		row = session.run(q, sub=subject_name, obj=object_name, rel=relation_name).single()
		return bool(row and (row["c"] or 0) > 0)


def upsert_document_with_sentence(*, source_url: str, title: str, text_excerpt: str) -> Dict[str, Any] | None:
	"""Create Document and a single Sentence node (doc-scoped) and return ids. Returns None if no driver."""
	driver = get_driver()
	if driver is None:
		return None
	q = """
	MERGE (d:Document {source_url:$url}) ON CREATE SET d.title=$title
	WITH d
	OPTIONAL MATCH (d)-[:HAS_SECTION]->(s0:Section)
	WITH d, coalesce(max(s0.order), 0) + 1 AS nextOrder
	MERGE (s:Section {doc_url:$url, order:nextOrder})
	MERGE (p:Paragraph {doc_url:$url, order:1})
	MERGE (sen:Sentence {doc_url:$url, order:1, text_excerpt:$text})
	MERGE (d)-[:HAS_SECTION]->(s)
	MERGE (s)-[:HAS_PARAGRAPH]->(p)
	MERGE (p)-[:HAS_SENTENCE]->(sen)
    RETURN elementId(d) AS doc_id, elementId(sen) AS sentence_id
	"""
	with driver.session() as session:
		row = session.run(q, url=source_url, title=title, text=text_excerpt).single()
		return {"doc_id": row["doc_id"], "sentence_id": row["sentence_id"], "source_url": source_url}


def link_sentence_mentions_entity(*, source_url: str, sentence_order: int, entity_name: str, confidence: float = 0.8) -> Dict[str, Any] | None:
	"""Create a MENTIONS edge from a sentence (by doc_url + order) to an Entity by name. Returns None if no driver."""
	driver = get_driver()
	if driver is None:
		return None
	q = """
	MATCH (sen:Sentence {doc_url:$url, order:$ord})
	MERGE (e:Entity {name:$name})
	MERGE (sen)-[:MENTIONS {confidence:$conf, via:'heuristic'}]->(e)
    RETURN elementId(sen) AS sentence_id, elementId(e) AS entity_id
	"""
	with driver.session() as session:
		row = session.run(q, url=source_url, ord=int(sentence_order), name=entity_name, conf=float(confidence)).single()
		return {"sentence_id": row["sentence_id"], "entity_id": row["entity_id"], "entity": entity_name}



from __future__ import annotations

from typing import Dict, List, Optional

from db.neo4j_client import get_driver


def find_films_by_slots(known: Dict[str, str]) -> List[str]:
	"""Return film entity names matching all provided SlotValue constraints.

	Requires a :Type {name:'Film'} and (:Entity)-[:INSTANCE_OF]->(:Type) pattern, and HAS_SLOT edges.
	"""
	driver = get_driver()
	if driver is None:
		return []
	q = """
	MATCH (e:Entity)-[:INSTANCE_OF]->(:Type {name:'Film'})
	WITH e
	OPTIONAL MATCH (e)-[hs:HAS_SLOT]->(sv:SlotValue)
	WITH e, collect({slot: sv.slot, value: sv.value}) AS slots
	RETURN e.name AS name, slots
	"""
	res: List[str] = []
	with driver.session() as session:
		for row in session.run(q):
			name = row["name"]
			slots = row["slots"] or []
			ok = True
			for k, v in known.items():
				if not any((s.get("slot") == k and s.get("value") == v) for s in slots):
					ok = False
					break
			if ok:
				res.append(name)
	return res


def get_slot_value(entity_name: str, slot: str) -> Optional[str]:
	"""Return the SlotValue.value for (entity_name, slot) if present."""
	driver = get_driver()
	if driver is None:
		return None
	q = """
	MATCH (e:Entity {name:$name})-[hs:HAS_SLOT]->(sv:SlotValue {slot:$slot})
	RETURN sv.value AS value
	"""
	with driver.session() as session:
		row = session.run(q, name=entity_name, slot=slot).single()
		return None if row is None else row["value"]



def get_entity_types(entity_name: str) -> List[str]:
	"""Return list of Type.name values for a given entity name."""
	driver = get_driver()
	if driver is None:
		return []
	q = """
	MATCH (e:Entity {name:$name})-[:INSTANCE_OF]->(t:Type)
	RETURN DISTINCT t.name AS type
	"""
	with driver.session() as session:
		rows = session.run(q, name=entity_name).data()
		return [str(r["type"]) for r in rows if r and r.get("type")]


def list_slots_for_type(type_name: str, limit: Optional[int] = None) -> List[str]:
	"""List distinct SlotValue.slot names observed for entities of a given type.

	Ordered by frequency descending.
	"""
	driver = get_driver()
	if driver is None:
		return []
	q = """
	MATCH (e:Entity)-[:INSTANCE_OF]->(t:Type {name:$type})
	MATCH (e)-[:HAS_SLOT]->(sv:SlotValue)
	RETURN sv.slot AS slot, count(*) AS freq
	ORDER BY freq DESC
	"""
	with driver.session() as session:
		rows = session.run(q, type=type_name).data()
		slots = [str(r["slot"]) for r in rows if r and r.get("slot")]
		return slots[:limit] if (limit is not None and limit > 0) else slots


def list_common_slots(limit: Optional[int] = None) -> List[str]:
	"""List distinct SlotValue.slot names across all entities, by frequency."""
	driver = get_driver()
	if driver is None:
		return []
	q = """
	MATCH (:Entity)-[:HAS_SLOT]->(sv:SlotValue)
	RETURN sv.slot AS slot, count(*) AS freq
	ORDER BY freq DESC
	"""
	with driver.session() as session:
		rows = session.run(q).data()
		slots = [str(r["slot"]) for r in rows if r and r.get("slot")]
		return slots[:limit] if (limit is not None and limit > 0) else slots



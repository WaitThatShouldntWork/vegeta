// Minimal Neo4j bootstrap for VEGETA
// Constraints and indexes (idempotent)

// Unique IDs for core entities
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.id IS UNIQUE;

// CVE identifiers should be unique
CREATE CONSTRAINT cve_id_unique IF NOT EXISTS
FOR (c:CVE)
REQUIRE c.id IS UNIQUE;

// Event date index for time-bounded queries
CREATE INDEX event_date_index IF NOT EXISTS
FOR (ev:Event)
ON (ev.date);

// Basic label indexes for performance (optional)
CREATE INDEX cve_id_index IF NOT EXISTS FOR (c:CVE) ON (c.id);
CREATE INDEX product_id_index IF NOT EXISTS FOR (p:Product) ON (p.id);



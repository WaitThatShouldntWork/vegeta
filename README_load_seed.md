# VEGETA Database Loader

Complete data engineering pipeline for setting up the VEGETA Neo4j knowledge graph.

## What it does

1. **Executes seed.cypher** - Creates the base graph structure with entities, relationships, and constraints
2. **Generates embeddings** - Creates semantic embeddings for all entities using Ollama's `nomic-embed-text` model
3. **Creates indexes** - Sets up vector indexes, fulltext indexes, and standard property indexes for performance
4. **Validates data** - Checks data integrity and reports comprehensive statistics

## Prerequisites

- Neo4j database running on `bolt://localhost:7687`
- Ollama running on `http://localhost:11434` 
- `nomic-embed-text` model available in Ollama (script will auto-pull if missing)

## Usage

### Basic usage (recommended)
```powershell
python load_seed.py --reset
```

### Advanced options
```powershell
# Skip embedding generation (faster, but no semantic search)
python load_seed.py --reset --skip-embeddings

# Skip index creation 
python load_seed.py --reset --skip-indexes

# Use custom seed file
python load_seed.py --seed-file "custom_seed.cypher" --reset

# Load without clearing existing data
python load_seed.py
```

## What gets created

### Node Types
- **Entity**: Films, People, Awards (with embeddings)
- **Type**: Film, Person, Year, Award
- **SlotValue**: Genre, AwardsSignal values
- **Document/Section/Paragraph/Sentence**: Provenance hierarchy
- **Fact**: Reified claims (e.g., "Skyfall won BAFTA")
- **Checklist/SlotSpec**: Procedural templates

### Indexes Created
- **Vector**: `entity_semantic_index` for similarity search on embeddings
- **Fulltext**: `entity_name_fulltext` for name/alias lookup
- **Standard**: Property indexes on id, name, url fields for performance

### Embeddings
Each entity gets:
- `sem_emb`: Semantic embedding (768-dim, L2 normalized) from name + aliases + description
- `embed_text`: Text used to generate the embedding
- `embed_timestamp`: When the embedding was created

## Expected Output

```
📁 Executing Cypher file: study/seed.cypher
✓ Cypher file executed successfully
⚡ Creating standard property indexes...
✓ Created/verified 5 standard indexes
🔍 Creating fulltext indexes...
✓ Created entity name fulltext index
🧠 Generating embeddings for entities...
Found 18 entities to process
  Processed 10/18 entities
✓ Generated embeddings for 18 entities
📊 Creating vector indexes...
✓ Created semantic embedding vector index
🔍 Validating data integrity...
✓ Data validation completed

============================================================
📊 DATABASE LOADING STATISTICS
============================================================
Entities processed: 18
Embeddings generated: 18
Indexes created: 7
Errors encountered: 0

📋 NODE COUNTS:
  Entity: 18
  Type: 4
  SlotValue: 8
  Document: 1
  Checklist: 1
  Fact: 1

🧠 EMBEDDING COVERAGE:
  Total entities: 18
  With embeddings: 18
  Coverage: 100.0%

📊 CONSTRAINTS: 8
📊 INDEXES: 7
============================================================
🎉 Database loading completed successfully!
```

## Troubleshooting

### Neo4j connection issues
- Ensure Neo4j is running: `neo4j start`
- Check credentials in `load_seed.py` (default: neo4j/password)
- Verify port 7687 is accessible

### Ollama issues
- Start Ollama: `ollama serve`
- Check available models: `ollama list`
- Pull embedding model manually: `ollama pull nomic-embed-text`

### Vector index warnings
- Vector indexes require Neo4j 5.0+
- Script will warn and continue without vector indexes on older versions
- Semantic search will fall back to brute-force similarity in `attempt1.py`

### Memory considerations
- Embedding generation uses ~30MB RAM per 1000 entities
- Vector indexes require additional memory for similarity search
- Consider `--skip-embeddings` for large datasets or limited memory

## Integration with attempt1.py

After running this script, `attempt1.py` will:
- ✅ Find entities with embeddings (no more warnings!)
- ✅ Use vector indexes for fast similarity search
- ✅ Fall back gracefully if embeddings are missing
- ✅ Use fulltext indexes for entity linking

The warnings you saw before will be eliminated because entities will have the expected `sem_emb` properties.

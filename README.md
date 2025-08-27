# VEGETA - Variational Evidence Graph, Estimating Temporal Activations

A sophisticated system for active inference using Bayesian decision theory, designed for complex conversational interactions and graph-based knowledge retrieval.

## Features

- **🧠 Bayesian Active Inference**: Uses Expected Information Gain (EIG) to decide between asking questions, searching for facts, or providing answers
- **🔗 Graph-Based Knowledge**: Leverages Neo4j knowledge graphs with semantic embeddings
- **🔄 Multi-Turn Conversations**: Maintains conversation state and belief carryover across turns
- **🎯 Smart Question Generation**: Uses LLMs to generate natural clarifying questions
- **📊 Uncertainty Quantification**: Tracks confidence and reasoning for transparent decision-making
- **🎮 20-Questions Gameplay**: Perfect for guessing games and interactive discovery

## Quick Start

### Prerequisites

1. **Neo4j Database**: Running locally on bolt://localhost:7687
2. **Ollama**: Running locally on http://localhost:11434 with these models:
   - `gemma:4b` (for question generation)
   - `nomic-embed-text` (for embeddings)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd VEGETA

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Set up the database (load seed data)
python load_seed.py
```

### Usage

#### Interactive CLI Mode (20-Questions Style)

```bash
# Start interactive session
python -m vegeta.interfaces.cli interactive
```

Example interaction:
```
You: I'm thinking of a Pierce Brosnan spy film from the 1990s
VEGETA [ASK]: Which specific film are you thinking of - GoldenEye or another Bond film from that era?
Confidence: 45%

You: It has a tank chase scene
VEGETA [ANSWER]: Based on your clues, you're thinking of GoldenEye (1995)!
Confidence: 87%
```

#### Single Query Mode

```bash
# Process one query
python -m vegeta.interfaces.cli query "I want action movies similar to Heat"
```

#### All CLI Commands

```bash
# Help and Information
python -m vegeta.interfaces.cli --help                    # Show main help
python -m vegeta.interfaces.cli interactive --help        # Interactive mode help
python -m vegeta.interfaces.cli query --help              # Query mode help
python -m vegeta.interfaces.cli benchmark --help          # Benchmark help

# Logging Control
python -m vegeta.interfaces.cli query "test"              # Clean output (warnings only)
python -m vegeta.interfaces.cli --verbose query "test"    # Show timing and info logs
python -m vegeta.interfaces.cli --debug query "test"      # Show all debug logs (very verbose)

# Interactive Mode (20-Questions Game)
python -m vegeta.interfaces.cli interactive               # Start interactive session
python -m vegeta.interfaces.cli interactive --config custom.yaml  # Use custom config

# Single Query Processing
python -m vegeta.interfaces.cli query "Hello"             # Simple query
python -m vegeta.interfaces.cli query "I want spy movies" # Movie query
python -m vegeta.interfaces.cli query "Pierce Brosnan films from 1990s" # Specific query
python -m vegeta.interfaces.cli query --config custom.yaml "test query"  # With custom config

# Benchmarking and Testing
python -m vegeta.interfaces.cli benchmark                 # Quick benchmark (default)
python -m vegeta.interfaces.cli benchmark --type minimal  # 1 test case (~30 sec)
python -m vegeta.interfaces.cli benchmark --type quick    # 2 cases per category
python -m vegeta.interfaces.cli benchmark --type single   # All single-turn tests
python -m vegeta.interfaces.cli benchmark --type multi    # All multi-turn tests
python -m vegeta.interfaces.cli benchmark --type full     # Complete test suite
python -m vegeta.interfaces.cli benchmark --type quick --verbose  # With detailed logs
python -m vegeta.interfaces.cli benchmark --save          # Save results (default)
python -m vegeta.interfaces.cli benchmark --type full --verbose --config custom.yaml  # Full test with custom config

# Alternative Benchmark Tool
python -m vegeta.interfaces.benchmark_cli --type quick    # Standalone benchmark CLI
python -m vegeta.interfaces.benchmark_cli --type full --save --verbose  # Full benchmark with all options
```

#### Command Reference

| Command | Description | Options |
|---------|-------------|---------|
| `interactive` | Start 20-questions style conversation | `--config`, `--verbose`, `--debug` |
| `query <text>` | Process single query | `--config`, `--verbose`, `--debug` |
| `benchmark` | Run automated tests | `--type`, `--save`, `--verbose`, `--debug`, `--config` |

#### Benchmark Types

| Type | Description | Test Cases | Estimated Time |
|------|-------------|------------|----------------|
| `minimal` | Ultra-fast validation | 1 test case | 30 seconds |
| `quick` | Fast validation | 2 per category | 5-10 minutes |
| `single` | All single-turn tests | 21 cases | 15-25 minutes |
| `multi` | All multi-turn tests | 12 conversations | 20-30 minutes |
| `full` | Complete test suite | 33 total cases | 30-45 minutes |

#### Configuration

```bash
# Using custom configuration
python -m vegeta.interfaces.cli --config /path/to/config.yaml interactive
python -m vegeta.interfaces.cli query --config custom.yaml "test"
python -m vegeta.interfaces.cli benchmark --config production.yaml --type full
```

#### Python API

```python
from vegeta import VegetaSystem, Config

# Initialize system
config = Config()  # or Config("custom.yaml")
system = VegetaSystem(config)

# Start session and process queries
session_id = system.start_session()
response = system.process_query(session_id, "I'm thinking of a sci-fi film from 1999")

print(f"Action: {response.action}")
print(f"Content: {response.content}")
print(f"Confidence: {response.confidence:.1%}")
```

## Architecture

### Core Components

```
src/
├── core/               # System orchestration and configuration
├── extraction/         # Entity extraction and semantic analysis
├── retrieval/          # Graph-based candidate retrieval
├── inference/          # Bayesian prior/posterior updates
├── session/            # Multi-turn conversation management
├── generation/         # Natural language question/answer generation
├── interfaces/         # CLI and web interfaces
└── utils/              # Database and LLM utilities
```

### Key Algorithms

1. **Entity Extraction**: LLM-based extraction of canonical terms and entities
2. **Semantic Retrieval**: Embedding-based anchor selection and subgraph expansion
3. **Bayesian Inference**: Prior construction and posterior updates using observed evidence
4. **Decision Making**: Expected Information Gain calculation for ASK vs SEARCH vs ANSWER
5. **Question Generation**: LLM-powered natural language question formulation

## Configuration

Configuration is managed through YAML files in the `config/` directory:

```yaml
database:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"

ollama:
  base_url: "http://localhost:11434"
  default_model: "gemma:4b"
  embedding_model: "nomic-embed-text"

system:
  defaults:
    k_anchors: 10
    tau_retrieval: 0.7
    alpha: 1.0    # Semantic weight
    beta: 0.5     # Structural weight
    gamma: 0.3    # Terms weight
```

## Multi-Turn Sessions

VEGETA maintains conversation state across turns:

- **Belief Carryover**: Posteriors from previous turns inform current priors
- **Conversation History**: Tracks questions asked and answers received
- **Adaptive Learning**: Adjusts strategy based on user responses
- **Session Management**: Handles timeouts and session persistence

## Knowledge Graph Schema

The system uses a flexible ontology (see `docs/ontology.md`):

- **Entities**: Films, People, Awards, etc.
- **SlotValues**: Compact attributes (Genre, Era, Country)
- **Facts**: Reified claims with provenance
- **Checklists**: Procedural templates for different tasks

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vegeta
```

### Adding New Domains

1. Extend the ontology in `docs/ontology.md`
2. Add domain-specific entities to seed data
3. Create domain checklists for task types
4. Update extraction patterns if needed

## Examples

### Movie Identification
```
User: "I'm thinking of a film. Try to guess it."
VEGETA: "Could you tell me what genre this film is?"
User: "It's a sci-fi action movie"
VEGETA: "Do you remember approximately when it was released?"
User: "Late 1990s"
VEGETA: "Are there any specific actors you remember?"
User: "Keanu Reeves"
VEGETA: "Based on your clues, you're thinking of The Matrix (1999)!"
```

### Movie Recommendation
```
User: "I want action movies similar to Heat"
VEGETA: "What specifically did you like about Heat - the heist elements, the cat-and-mouse dynamic, or the realistic action?"
User: "The realistic action and complex characters"
VEGETA: "Based on those preferences, I'd recommend Ronin (1998) - it has similar realistic action sequences and complex character dynamics."
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**: Ensure Neo4j is running and credentials are correct
2. **Ollama Models Missing**: Run `ollama pull gemma:12b` and `ollama pull nomic-embed-text`
3. **Low Confidence Responses**: Check if embeddings were generated during database setup
4. **Import Errors**: Ensure you've installed in development mode with `pip install -e .`

### Logging

Set log level in config or environment:
```bash
export PYTHONPATH=src
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from vegeta import VegetaSystem, Config
system = VegetaSystem(Config())
"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use VEGETA in your research, please cite:

```bibtex
@misc{vegeta2024,
  title={VEGETA: Bayesian Active Inference for Interactive Knowledge Discovery},
  author={VEGETA Project},
  year={2024},
  url={https://github.com/your-org/vegeta}
}
```

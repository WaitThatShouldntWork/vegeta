# VEGETA - Variational Evidence Graph, Estimating Temporal Activations

A  **Bayesian Active Inference System** implementing complete **predictive coding** architecture for complex conversational interactions and graph-based knowledge retrieval.

VEGETA is a working Bayesian active inference prototype with:
- ✅ **Complete Predictive Coding Pipeline**: Generative model → likelihood → posterior updates
- ✅ **Three-Channel Predictions**: Semantic, structural, and terms prediction channels
- ✅ **Active Inference Engine**: EIG-based decision making with ASK/SEARCH/ANSWER actions
- ✅ **Comprehensive Benchmarking**: 33 test cases with evaluation metrics
- ✅ **Multi-Turn Session Management**: Belief carryover and context awareness
- ✅ **Graph-Based Knowledge Retrieval**: Neo4j integration with 21 nodes and 100% embedding coverage

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
   - `qwen3:8b` (for question generation, default)
   - `gemma:4b` or `deepseek-r1:8b` (alternative models)
   - `nomic-embed-text:latest` (for embeddings)

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
python -m vegeta.interfaces.cli interactive --verbose     # Interactive with detailed timing and debug info
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

# Using console scripts (after pip install -e .)
vegeta interactive
vegeta query "test query"
vegeta-benchmark --type quick
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
vegeta/
├── core/               # System orchestration and configuration
├── extraction/         # Entity extraction and semantic analysis
├── retrieval/          # Graph-based candidate retrieval
├── inference/          # Complete predictive coding pipeline
│   ├── active_inference_engine.py  # EIG-based decision making
│   ├── generative_model.py         # Three-channel predictions
│   ├── likelihood_computer.py      # Evidence integration
│   ├── prior_builder.py           # Multi-source prior construction
│   └── types.py                   # Type definitions
├── session/            # Multi-turn conversation management
├── generation/         # Natural language question/answer generation
├── interfaces/         # CLI and web interfaces
├── testing/            # Comprehensive benchmark suite
└── utils/              # Database and LLM utilities
```

### Key Algorithms

1. **Complete Predictive Coding**: Three-channel generative model (semantic, structural, terms)
2. **Entity Extraction**: LLM-based extraction of canonical terms and entities
3. **Semantic Retrieval**: Embedding-based anchor selection and subgraph expansion
4. **Bayesian Active Inference**: Full likelihood computation with evidence integration
5. **Multi-Source Prior Construction**: Step priors, slot priors, and conversation history
6. **Expected Information Gain**: 2-step EIG planning for optimal decision making
7. **Adaptive Question Generation**: Context-aware LLM question formulation
8. **Session State Management**: Belief carryover with inertia parameter ρ

## Configuration

Configuration is managed through YAML files in the `config/` directory:

```yaml
database:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"
  database: "neo4j"

ollama:
  base_url: "http://localhost:11434"
  default_model: "qwen3:8b"
  embedding_model: "nomic-embed-text:latest"

system:
  defaults:
    k_anchors: 10
    M_candidates: 20
    hops: 2
    tau_retrieval: 0.7
    tau_posterior: 0.7
    alpha: 1.0         # Semantic weight
    beta: 0.5          # Structural weight
    gamma: 0.3         # Terms weight
    sigma_sem_sq: 0.3
    sigma_struct_sq: 0.2
    sigma_terms_sq: 0.2
    N_terms_max: 15
    N_expected: 20
    small_set_threshold: 3
    small_set_blend: 0.5
    lambda_missing: 0.30
    d_cap: 40
    lambda_hub: 0.02

session:
  timeout: 3600       # 1 hour
  max_turns: 20
  inertia_rho: 0.7

  # Context window management
  max_context_tokens: 2048
  summary_tokens: 256
  context_window_turns: 5
  compression_ratio: 0.3

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Multi-Turn Sessions & Predictive Coding

VEGETA implements a complete **predictive coding architecture** with sophisticated Bayesian inference for multi-turn conversations:

### 🧠 Complete Predictive Coding Pipeline

VEGETA's core is a fully operational Bayesian active inference system with three-channel predictive coding:

1. **Generative Model (G)**: Predicts expected observations from hidden states
   - Semantic predictions using embedding similarity
   - Structural predictions using checklist requirements
   - Terms predictions using subgraph analysis

2. **Likelihood Computation**: Three-channel distance computation with proper normalization
   - Semantic likelihood using embedding similarity
   - Structural likelihood using checklist expectations
   - Terms likelihood with subgraph analysis and penalties

3. **Prior Construction**: Multi-source prior building
   - Step priors from procedure tracking
   - Slot priors from candidate subgraph analysis
   - Conversation history integration with inertia parameter ρ

4. **Active Inference Engine**: EIG-based decision making
   - 2-step Expected Information Gain planning
   - ASK/SEARCH/ANSWER action selection
   - Posterior updates with confidence scoring

### 🎯 MOMDP Architecture (In Development)

VEGETA is being formalized as a **Mixed Observability Markov Decision Process** for principled action selection:

**State Factorization:**
- **Observed State (x)**: Checklist, procedure step, filled slots, retrieved facts
- **Hidden State (y)**: Unfilled slot values, subgraph beliefs, novelty indicators

**Core Components:**
- **T(s'|s,a)**: Transition dynamics with factorized observed/hidden state evolution
- **O(o|x',y',a)**: Action-conditioned observation model with slot confusion rates
- **R(x,y,a)**: Reward model balancing accuracy, cost, and time efficiency
- **Belief Update**: `b_{t+1}(y') ∝ O(o|x',y',a) * Σ_y T_y(y'|x,y,a) * b_t(y)`

This formalization provides:
- Rigorous EIG calculations using proper observation likelihoods
- Principled multi-turn belief carryover through state transitions
- Clear separation of concerns between observed facts and uncertain hypotheses

### 🎯 Advanced Session Management

VEGETA maintains sophisticated conversation state across turns:

- **Belief Carryover**: Posteriors from previous turns inform current priors through tempered carryover
- **Observation Persistence**: Stores utterance observations (u) to influence next turn's priors
- **Context Window Management**: Smart token management with conversation compression
- **Entity Memory**: Remembers confirmed entities to avoid repetitive questions
- **Session Persistence**: Handles timeouts, turn limits, and session recovery

### 📊 Bayesian State Management

Advanced Bayesian filtering with predictive coding:

```python
# Complete predictive coding pipeline
g(v) → u'_sem, u'_struct, u'_terms  # Generative predictions
p(u|z) = likelihood computation     # Evidence integration
p(z|u) = posterior updates         # Belief refinement
EIG = expected information gain     # Decision optimization

# Multi-turn belief evolution
p_{t+1}(z) ∝ (q_t(z))^ρ × base_prior × context_boost × procedure_prior

# Where:
# - ρ ∈ (0,1]: inertia parameter (0.7 = balanced, 1.0 = sticky)
# - context_boost: utterance similarity + entity continuity (0.3-0.9)
# - procedure_prior: checklist-driven expectations
# - q_t(z): posterior beliefs from previous turn
```

**Key Features:**
- **Complete Predictive Coding**: Full generative model → likelihood → posterior pipeline
- **Three-Channel Predictions**: Semantic, structural, and terms prediction channels
- **Procedure-Driven Reasoning**: Checklist-based task execution with slot filling
- **Uncertainty Quantification**: Confidence scores with calibration and entropy analysis
- **Adaptive Question Generation**: Context-aware LLM question formulation

## Knowledge Graph Schema

The system uses a flexible ontology (see `docs/ontology.md`):

- **Entities**: Films, People, Awards, etc.
- **SlotValues**: Compact attributes (Genre, Era, Country)
- **Facts**: Reified claims with provenance
- **Checklists**: Procedural templates for different tasks

## Development

### Running Tests & Benchmarks

```bash
# Run all unit tests
pytest

# Run with coverage
pytest --cov=vegeta

# Run comprehensive benchmark suite
python -m vegeta.interfaces.cli benchmark --type full --verbose --save

# Run quick validation benchmark
python -m vegeta.interfaces.cli benchmark --type quick

# Using console scripts
vegeta-benchmark --type quick --verbose
```

### Benchmark Types

- **minimal**: Ultra-fast validation (1 test case, 30 seconds)
- **quick**: Fast validation (2 cases per category, 5-10 minutes)
- **single**: All single-turn tests (21 cases, 15-25 minutes)
- **multi**: All multi-turn tests (12 conversations, 20-30 minutes)
- **full**: Complete test suite (33 total cases, 30-45 minutes)

### Adding New Domains

1. Extend the ontology in `docs/ontology.md`
2. Add domain-specific entities to seed data
3. Create domain checklists for task types
4. Update extraction patterns if needed
5. Run benchmarks to validate new domain performance

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

## Current Development Phase

VEGETA is currently in **Phase 2: Memory-Augmented Intelligence** of its roadmap:

### ✅ **Phase 1: Core Predictive Coding** - COMPLETED
- Complete predictive coding pipeline operational
- Bayesian active inference with EIG-based decisions
- Comprehensive benchmarking and evaluation

### 🟡 **Phase 2: Memory-Augmented Intelligence** - CURRENT
- **MOMDP Formalization**: Converting to formal Mixed Observability Markov Decision Process
- Episodic case bank implementation
- External search and tool integration
- Sleep-like graph maintenance
- Multi-agent collaborative intelligence

### 🟢 **Phase 3: Advanced Capabilities** - FUTURE
- Neuroplasticity-inspired meta-learning
- Real-time adaptation
- Federated learning across instances

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run benchmarks to ensure performance
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use VEGETA in your research, please cite:

```bibtex
@misc{vegeta2024,
  title={VEGETA: Variational Evidence Graph, Estimating Temporal Activations},
  author={VEGETA Project},
  year={2024},
  url={https://github.com/waitThatShouldntWork/vegeta}
}
```

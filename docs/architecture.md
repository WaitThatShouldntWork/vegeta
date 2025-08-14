# VEGETA System Architecture

This document provides a visual overview of the Value Estimation Gain Evaluation Tool & Analysis (VEGETA) system architecture.

## System Architecture Diagram

```mermaid
graph TB
    %% User Interface Layer
    UI[/"User Interface"/]
    CLI["CLI Demo<br/>--domain cyber --cve CVE-2025-XXXX"]
    WebUI["Web Panel<br/>(Optional)"]
    
    %% Core Decision Engine
    subgraph "EIG Decision Engine"
        State["State Tracker<br/>decider/state.py"]
        Confidence["Confidence Calculator<br/>decider/confidence.py"]
        EIG["EIG Calculator<br/>decider/eig.py"]
        Choose["Action Chooser<br/>decider/choose.py"]
        
        State --> Confidence
        Confidence --> EIG
        EIG --> Choose
    end
    
    %% Actions
    subgraph "Action Types"
        Answer["Answer Action<br/>Confidence - Risk"]
        Ask["Ask Action<br/>Question to User"]
        Search["Search Action<br/>Retrieve Information"]
    end
    
    %% Working Set & Graph Processing
    subgraph "Graph Processing"
        WorkingSet["Working Set Builder<br/>ingest/working_set.py<br/>≤25k nodes, ≤250k rels"]
        GDS["AuraDS GDS<br/>fastRP/PageRank priors"]
        Encoder["GNN Encoder (optional later)<br/>q_φ(v|G,D)"]
        Decoder["Generative Decoder<br/>p_θ(u|v)"]
        
        WorkingSet --> GDS
        GDS --> Confidence
        WorkingSet --> Encoder
        Encoder --> Decoder
    end
    
    %% Knowledge Graph
    subgraph "Neo4j Knowledge Graph"
        KG[("Knowledge Graph<br/>Entities, Relations<br/>Procedures, Policies")]
        CVE["CVE Nodes<br/>CVSS, EPSS"]
        Products["Product/Vendor<br/>Nodes"]
        Techniques["MITRE ATT&CK<br/>Techniques"]
        Events["Events<br/>Exploited in Wild"]
    end
    
    %% Domain Packs
    subgraph "Domain Packs"
        CyberPack["Cybersecurity Pack<br/>domain_packs/cyber/"]
        TransitPack["Transit Pack<br/>(Future)"]
        ContractPack["Contracting Pack<br/>(Future)"]
        
        subgraph "Cyber Pack Components"
            CyberConfig["config.yaml"]
            CyberMappings["mappings.py"]
            CyberQuestions["questions.yaml<br/>12-15 canonical questions"]
        end
    end
    
    %% Data Ingestion
    subgraph "Data Ingestion"
        ETL["ETL Pipeline<br/>ingest/cyber_etl.py"]
        NVD["NVD CVE Feed"]
        CPE["CPE Dictionary"]
        KEV["CISA KEV"]
        EPSS["EPSS Scores"]
        ATTACK["MITRE ATT&CK"]
    end
    
    %% Retrieval System
    subgraph "Retrieval System"
        RetBase["Retrieval Base<br/>retrieval/base.py"]
        LocalRet["Local Snapshot<br/>retrieval/local_snapshot.py"]
        HTTPRet["HTTP Clients<br/>retrieval/http_clients.py"]
    end
    
    %% Evaluation & Testing
    subgraph "Evaluation"
        Scenarios["Test Scenarios<br/>eval/scenarios_cyber.jsonl"]
        UserSim["User Simulator<br/>Noisy responses"]
        Metrics["Metrics<br/>Accuracy, Questions, Latency"]
        EMGSeed["EMG Seed + Episodes<br/>(steps, asks, searches)"]
        RunEval["eval/run_eval.py"]
    end
    
    %% Schema & Validation
    subgraph "Schema & Metamodel"
        Schema["Metamodel<br/>schema/metamodel.yaml"]
        Linter["Schema Linter<br/>schema/lint_schema.py"]
        Bootstrap["Neo4j Bootstrap<br/>schema/bootstrap.cypher"]
    end
    
    %% Connections
    UI --> CLI
    UI --> WebUI
    CLI --> Choose
    WebUI --> Choose
    
    Choose --> Answer
    Choose --> Ask
    Choose --> Search
    
    Search --> RetBase
    RetBase --> LocalRet
    RetBase --> HTTPRet
    
    WorkingSet --> KG
    KG --> CVE
    KG --> Products
    KG --> Techniques
    KG --> Events
    
    ETL --> KG
    NVD --> ETL
    CPE --> ETL
    KEV --> ETL
    EPSS --> ETL
    ATTACK --> ETL
    
    CyberPack --> CyberConfig
    CyberPack --> CyberMappings
    CyberPack --> CyberQuestions
    
    CyberMappings --> ETL
    CyberQuestions --> Choose
    
    Schema --> Bootstrap
    Schema --> Linter
    Bootstrap --> KG
    
    RunEval --> Scenarios
    RunEval --> UserSim
    RunEval --> Metrics
    RunEval --> EMGSeed
    %% End-to-end active loop (high-level)
    subgraph "Active Loop (End-to-End)"
        A0["Uncertainty & Slot Selection\n(EIG over slots)"]
        A1["Question Generation\n(Template or LLM via Ollama)"]
        A2["Search Wikipedia/Wikidata\n(retrieval/wikipedia.py)"]
        A3["Parse & Write Back\n(:Fact + HAS_SOURCE + Lexical)"]
        A4["Update Belief / Entropy\n(reduce uncertainty)"]
    end

    Choose --> A0
    A0 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> Choose
    
    %% Data flow for decision making
    KG -.-> WorkingSet
    WorkingSet -.-> State
    Decoder -.-> EIG
    GDS -.-> Confidence
    LocalRet -.-> State
    
    %% Styling
    classDef engine fill:#e1f5fe
    classDef data fill:#f3e5f5
    classDef action fill:#e8f5e8
    classDef eval fill:#fff3e0
    
    class State,Confidence,EIG,Choose engine
    class KG,CVE,Products,Techniques,Events data
    class Answer,Ask,Search action
    class Scenarios,UserSim,Metrics,RunEval eval
```

## Key Components

### Core Decision Engine
- **State Tracker**: Maintains current knowledge state and evidence
- **Confidence Calculator**: Computes confidence scores for potential answers
- **EIG Calculator**: Calculates Expected Information Gain for each action
- **Action Chooser**: Selects optimal action (Answer/Ask/Search) based on EIG and costs

### Graph Processing
- **Working Set Builder**: Creates bounded subgraphs for processing
- **GNN Encoder**: Neural network encoder for graph representations
- **Generative Decoder**: Predicts observations from latent states

### Knowledge Graph (Neo4j)
- **Entities**: CVEs, Products, Vendors, Techniques
- **Relations**: Whitelisted predicates (IS_A, AFFECTS, MITIGATES, etc.)
- **Procedures**: Workflow steps and decision branches
- **Policies**: Cost models and risk assessments

### Domain Packs
- **Cybersecurity Pack**: CVE triage and vulnerability assessment
- **Future Packs**: Public Transit, Open Contracting (planned)
- **Components**: Configuration, mappings, canonical questions

### Data Sources
- **NVD**: National Vulnerability Database CVE feed
- **CPE**: Common Platform Enumeration dictionary
- **CISA KEV**: Known Exploited Vulnerabilities catalog
- **EPSS**: Exploit Prediction Scoring System
- **MITRE ATT&CK**: Tactics, techniques, and procedures

### Evaluation Framework
- **Test Scenarios**: Ground-truth datasets for validation
- **User Simulator**: Automated testing with realistic user responses
- **Metrics**: Accuracy, efficiency, and performance measurements

## Data Flow

1. **Ingestion**: External feeds → ETL → Knowledge Graph
2. **Working Set**: Task context → Bounded subgraph extraction
3. **Encoding**: Graph + Evidence → Neural representations
4. **Decision**: EIG calculation → Action selection
5. **Execution**: Answer/Ask/Search based on optimal choice
6. **Update**: New information → State update → Next iteration

## Design Principles

- **Bayesian Framework**: Principled uncertainty quantification
- **Active Inference**: Information-seeking behavior
- **Modular Architecture**: Pluggable domain packs and retrieval systems
- **Scalable Processing**: Bounded working sets and efficient algorithms
- **Evaluation-Driven**: Comprehensive testing and metrics

## Learning loop (simulator + policy)

- Policy: small trainable chooser over actions (Answer | Ask | Search). Start as a contextual bandit (softmax over features), not an RNN.
- Features: domain signals (e.g., `cvss_norm`, `epss`, `kev_flag`), belief entropy, simple priors (centrality), resolved slots, and cheap EIG estimates.
- Simulator (noisy user): when the policy picks Ask(slot), the env samples a yes/no from that slot's ground-truth with a small error rate, updates belief via our Bayesian layer, and returns a step reward. Episode ends on Answer or max-steps.
- Questions source: `domain_packs/<domain>/questions.yaml` defines slots/text; the agent picks the next question by highest EIG among eligible slots.


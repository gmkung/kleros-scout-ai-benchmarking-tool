# Kleros Scout - AI-curation benchmarking tool

This repository contains tools for evaluating contract metadata tagging accuracy, with a specific focus on comparing predictions against ground truth data established from the decentralized curation process from Kleros Scout (klerosscout.eth.limo).

## Overview

The project consists of two main components:
1. An evaluation script that measures the accuracy of contract metadata predictions
2. A reference implementation using the [Sonar API by Perplexity](https://docs.perplexity.ai/home) to generate predictions

## Evaluation Methodology

The evaluation process is designed to assess how well AI models can replicate human-curated contract metadata across different types of fields:

### Named Entity Recognition (NER) Elements
Fields requiring precise identification with some flexibility:
- **Project Name**: The official name of the project/protocol
  - Evaluated using F1 score and Levenshtein distance
  - Allows for minor variations (e.g., "Uniswap Protocol" vs "Uniswap")
- **Public Name Tag**: Token symbol or common identifier
  - Must match standard formats (e.g., "WETH", "UNI-V2")
  - Evaluated with exact matching and near-match tolerance

### Exact Match Elements
Fields requiring precise accuracy:
- **Contract Address**: Must be in CAIP-10 format
  - Example: "eip155:1:0x..." for Ethereum mainnet
  - No flexibility in matching - must be exact
- **UI/Website Link**: Official project URL
  - Must be valid and accessible
  - Normalized before comparison (handles www./https variations)

### Semantic Elements
Fields requiring meaning preservation:
- **Public Note**: Project description
  - Evaluated using cosine similarity
  - Focuses on semantic meaning rather than exact wording
  - Threshold of 0.85 for acceptable similarity
  - Checked for policy violations and accuracy

### Scoring Metrics

1. **Exact Match Rate**
   - Percentage of perfectly matching fields
   - Used for Contract Address and UI/Website Link

2. **NER Metrics**
   - Precision: Accuracy of identified entities
   - Recall: Completeness of identified entities
   - F1 Score: Harmonic mean of precision and recall

3. **Semantic Similarity**
   - Cosine similarity for text fields
   - Accounts for meaning preservation
   - Handles variations in description style

## Reference Implementation

The Perplexity implementation demonstrates:
- Chain-aware metadata generation
- Format validation
- Error handling
- Rate limiting
- Multi-chain support

## Evaluation Script (model-evaluation.py)

The evaluation script compares predicted contract metadata against ground truth data using multiple metrics:

### Fields Evaluated
- Contract Address (exact match)
- Project Name (NER evaluation)
- Public Name Tag (NER evaluation)
- UI/Website Link (exact match)
- Public Note (semantic evaluation)

### Evaluation Metrics
- **Exact Fields**: Direct string matching for addresses and URLs
- **NER Fields**: F1 score, precision, and recall with Levenshtein similarity
- **Semantic Fields**: Cosine similarity for descriptive text

### Usage
```bash
python eval-script-2.py
```

The script will:
1. Load ground truth and prediction data from JSONL files
2. Compute various similarity metrics
3. Generate a detailed evaluation report
4. Save results to `evaluation_results.json`

## Reference Implementation (perplexity-chain-data.py)

A reference implementation using the Perplexity API to generate contract metadata predictions.

### Features
- CAIP-10 address format support
- Multi-chain support (Ethereum, Optimism, BSC, etc.)
- Configurable API parameters
- Robust error handling and validation

### Usage
```bash
python prediction_setups/perplexity-chain-data.py
```

### Environment Setup
Create a `.env` file with:
```
PERPLEXITY_API_KEY=your_api_key_here
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

### Ground Truth Data (JSONL)
```json
{
    "Contract Address": "eip155:1:0x...",
    "Project Name": "Project Name",
    "Public Name Tag": "Token Symbol",
    "UI/Website Link": "https://...",
    "Public Note": "Description"
}
```

### Directory Structure
```
├── data/
│   ├── ground-truth/
│   │   └── data-set1.jsonl
│   └── predictions/
│       └── data-set1.jsonl
├── prediction_setups/
│   └── perplexity-chain-data.py
├── eval-script-2.py
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.7+
- Dependencies listed in requirements.txt
- Perplexity API key for reference implementation

## License
MIT
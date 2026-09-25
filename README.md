
# Centralized Task Planning and Dynamic Device Collaboration in LLM-based Multi-Agent Smart Home Systems

This repository contains the implementation and benchmark for my Bachelor Thesis: **"Centralized Task Planning and Dynamic Device Collaboration in LLM-based Multi-Agent Smart Home Systems"**

## 📂 Documentation Guide

For detailed instructions, please refer to the specific documentation in each module:

*  **Benchmark**: For dataset details and evaluation metrics, please read the **[Benchmark README](./benchmark/README.md)**.
*  **System Implementation**: For installation steps, environment setup, and execution guide, please read the **[System Implementation README](./system_implementation/README.md)**.

## File Structure
```
bachelorarbeit/                   
├── README.md                       # This file
├── benchmark/                      # Evaluation Dataset
│   ├── benchmark_data.json
│   └── README.md
└── system_implementation/          # Source Code
    ├── requirements.txt            # Dependencies
    ├── smart_home_langgraph.py     # Main system implementation
    ├── smart_home_langgraph.ipynb  # Jupyter notebook version
    ├── run_benchmark.py            # Benchmark execution script
    └── logs/                       # Generated execution logs
        ├── README.md                
        ├── simple_01.txt           # Sample logs
        ├── moderate_01.txt
        └── complex_01.txt
```

## Technical Details

* Orchestration: LangGraph 1.0.5

* LLM Engine: Ollama (Version 0.13.5+)

* Model: gemma2:latest (9B parameters)

* Development Environment: Python 3.11 on MacBook Air (M4)
  
# Bachelor Thesis for Zeyi Sun

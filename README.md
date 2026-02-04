# StatABench

> **The Official Code Repository for StatABench**

This repository contains the implementation and benchmarking scripts for **StatABench**.

## 📂 Repository Structure

* `mcp_scripts/`: Scripts for running the MCP agent pipeline.
* `agent_scripts/`: Scripts for other agent frameworks (AutoGen, CrewAI, etc.).
* `evaluation/`: Evaluation scripts to score model outputs.
* `datasets83/`: (Implicit) Directory containing dataset CSVs.


## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed and the necessary dependencies. You can likely install them via:

```bash
pip install -r requirements.txt

```

### API Configuration

Before running any agents, you must configure your LLM credentials. Create or update the `keys.json` file in the root directory with your API details:

```json
{
  "deepseek": {
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your-api-key-here",
    "model": "deepseek-v3"
  },
  "qwen": {
    "base_url": "...",
    "api_key": "...",
    "model": "qwen-plus"
  }
}

```


## 🤖 Running Benchmarks

### 1. MCP Agent Pipeline

To test an LLM using the Model Context Protocol (MCP) pipeline, follow these steps:

**Step 1: Start the MCP Server**

```bash
python ./mcp_scripts/mcp_server.py

```

**Step 2: Run the Test Pipeline**
Open a new terminal and execute the pipeline script:

```bash
python ./mcp_scripts/mcp_pipeline.py \
  --model deepseek \
  --begin_index 0 \
  --input_path ./data/statabench.json \
  --output_path ./data/result/mcp_deepseek.json

```

**Parameters:**

* `--model`: The model to test.
* `--begin_index`: The index to start processing from (useful for resuming).
* `--input_path`: Path to the source dataset file.
* `--output_path`: Path to save the results. The script will append a new column (e.g., `mcp_response_{llm}`) to this file.

### 2. Multi-Agent Frameworks

You can also evaluate LLMs using other agent frameworks located in the `agent_scripts` folder. Supported frameworks include **AutoGen**, **CrewAI**, **Qwen-Agent**, and **SmolAgents**.

**Example: Running AutoGen with DeepSeek**

```bash
python ./agent_scripts/agent_autogen.py \
  --models deepseek \
  --begin_index 0 \
  --input_path ./data/statabench.json \
  --output_path ./data/result/autogen_deepseek.json

```

- The output file will contain a new column `autogen_deepseek` with the model's responses.

## 📊 Evaluation

Once the generation process is complete, you can evaluate the model's responses against the ground truth.

Run the evaluation script pointing to your result file:

```bash
python ./evaluation/eval.py \
  --input_path ./data/result/autogen_deepseek.json \
  --response_col autogen_deepseek

```

**Parameters:**

* `--input_path`: The file containing the model's generated responses.
* `--response_col`: The specific column name in the JSON file where the model's answers are stored (e.g., `autogen_deepseek`, `crewai_gpt4`).
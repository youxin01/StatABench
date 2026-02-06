# StatABench: Dataset and Framework for Evaluating Statistical Analysis Capabilities of LLMs

The official repository for **StatABench**. The benchmark is divided into two distinct tracks: **Stat-Closed** (Structured Benchmark) and **Stat-Open** (Open-ended Modeling).

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed and the necessary dependencies.

```bash
pip install -r requirements.txt

```

### API Configuration

Before running any agents, you must configure your LLM credentials. Create or update the `keys.json` file in the root directory.

**Example `keys.json` structure:**

```json
{
  "deepseek": {
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "your-api-key-here",
    "model": "deepseek-v3"
  },
  "qwen": {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "your-api-key-here",
    "model": "qwen-plus"
  }
}

```

---

## Stat-Closed

Stat-Closed evaluates LLMs on specific statistical queries using various agent frameworks.

### 1. MCP Agent Pipeline

To test an LLM using the [LangChain MCP](https://github.com/langchain-ai/langchain), follow these two steps:

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
  --input_path ./data/stat-closed.json \
  --output_path ./data/result/mcp_deepseek.json

```

**Arguments:**
* `--model`  The model key defined in `keys.json` (e.g., `deepseek`).
* `--begin_index`  The index to start processing from (useful for resuming interrupted runs).
* `--input_path`  Path to the source dataset file.
* `--output_path`  Path to save results. The script appends a new column (e.g., `mcp_response_{llm}`). 

### 2. Other Agent Frameworks

You can also evaluate LLMs using other popular agent frameworks located in the `agent_scripts` folder.
**Supported Frameworks:** AutoGen, CrewAI, Qwen-Agent, and SmolAgents.

**Example: Running AutoGen with DeepSeek**

```bash
python ./agent_scripts/agent_autogen.py \
  --models deepseek \
  --begin_index 0 \
  --input_path ./data/stat-open.json \
  --output_path ./data/result/autogen_deepseek.json

```

* **Note:** The output file will contain a new column (e.g., `autogen_deepseek`) with the model's responses.

### Evaluation (Stat-Closed)

Once the generation process is complete, evaluate the model's responses against the ground truth.

```bash
python ./evaluation/eval.py \
  --input_path ./data/result/autogen_deepseek.json \
  --response_col autogen_deepseek

```

**Arguments:**

* `--input_path`: The JSON file containing the model's generated responses.
* `--response_col`: The specific column name where the model's answers are stored (e.g., `autogen_deepseek`, `crewai_gpt4`).

---

## Stat-Open

**Stat-Open** focuses on real-world, open-ended statistical modeling problems. The core benchmark data is located in `data/stat-open.json`, also the dataset that the every problem needs can be found at [![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/ADUIDUIDUIi/Stat-Open-Datasets)


### Dataset Structure

Each entry in `stat-open.json` contains the following fields:

* **`background`**: Background introduction of the problem.
* **`problem_requirement`**: Specific requirements and questions to answer.
* **`dataset_path`**: Dataset name of the associated dataset files.
* **`dataset_description`**: Explanation of the dataset structure and source.
* **`variable_description`**: Detailed description of variables within the dataset.
* **`addendum`**: Appendix information or extra context.
* **`role`**: role description used by the Judger.

### Modeding Agent

We  test two advanced agent frameworks for this track:

1. **[MathModelAgent](https://github.com/jihe520/MathModelAgent)**
2. **[LLM-MM-Agent](https://github.com/usail-hkust/LLM-MM-Agent)**

Please follow their official repositories for instructions on how to set up and run tasks on the Stat-Open dataset.

### LLM-as-a-Judge Evaluation

After generating the analysis reports (in Markdown format), use our automated judger to evaluate the quality.

```bash
python judger-open/main_judge_stat.py \
  --model gpt-4o-mini \
  --problem cumcm2012c \
  --paper_path ./path_to_your_md_file.md

```

**Arguments:**

* `--model`: The judge model to use (e.g., `gpt-4o-mini`).
* `--problem`: The specific problem ID (e.g., `cumcm2012c`) matching the entry in the dataset.
* `--paper_path`: Path to the generated solution report (`.md` file).
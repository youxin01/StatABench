# env base1
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio
import sys
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import pandas as pd
from tqdm import tqdm

# # 1. First, set the event loop policy (must be done before any async operations)
# if sys.platform == "win32":
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

agent_map = {
    "qwen72": "qwen2.5-72b-instruct",
    "qwen32": "qwen2.5-32b-instruct",
    "qwen8": "qwen3-8b",
    "gpt4omini": "gpt-4o-mini",
    "deepseek": "deepseek-v3",
    "llama3.1-8b": "llama3.1-8b",
    "llama3.1-70b": "llama3.1-70b"
}

agent_name = "qwen8"
# model = ChatOpenAI(
#     model=agent_map[agent_name],
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     api_key=,
#     base_url="https://api.ai-gaochao.cn/v1",
# )

model = ChatOpenAI(
    model="qwen3-8b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={
        "enable_thinking": False,
        "return_reasoning": False
    }
)

# agent_name = "llama3.1-8b"
# model = ChatOpenAI(
#     model=agent_map[agent_name],
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     api_key="sk-XH5DQOWH964IdSnW4723Ef86D3844d37AeFfAeA569Cd9fAd",
#     base_url="http://localhost:54754/v1"
# )

current_dir = os.path.dirname(os.path.abspath(__file__))
math_server_path = os.path.join(current_dir, "mcp_server.py")

server_params = StdioServerParameters(
    command="python",
    args=[math_server_path],
)

system = """
You are a professional statistics analyst, and you can answer the user's questions with or without using tools.
"""

mcp_prompt = """
Your task is to answer the following question. Please follow the instructions carefully:

## Question

{questions}

## Instructions
1. Analyze the question and determine whether you can answer it.
2. There are many useful tools available, and you may use them to answer the user's questions (please use tools when necessary).
3. If the question is a multiple-choice or true/false question, you should strictly follow the required answer format and do not provide explanations.
4. For other questions, provide a concise and complete answer.
5. Finally, the output **must always** follow this format **(do not omit <>):**
    The answer is <your answer>.
"""

def get_mcp_prompt(row):
    if row["code"] == 0:
        eval_prompt = mcp_prompt.format(questions=row["question"])
    else:
        if row["dataset"] != 0:
            if not str(row["dataset"]).endswith(".csv"):
                dataset = str(row["dataset"]) + ".csv"
            else:
                dataset = str(row["dataset"])
            dataset_path = f"./datasets83/{dataset}"
            ques = (
                row["question"]
                + " The relevant dataset is located at: "
                + f'"{dataset_path}"'
                + "\n"
            )
        else:
            ques = row["question"]
        eval_prompt = mcp_prompt.format(questions=ques)
    return eval_prompt


datas = pd.read_json(
    f"./zz2_limit12/final_mcp_res_{agent_name}.json",
    encoding="utf-8"
)

data_tmp = datas.copy()
# output_file = f"./all_conversations_{agent_name}.txt"
output_file2 = f"./zz2_limit12/final_mcp_res_{agent_name}.json"

async def main():
    try:
        print("Connecting to server...")
        async with stdio_client(server_params) as (read, write):
            print("Server connected successfully, creating session...")
            print(math_server_path)
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("Loading tools...")
                tools = await load_mcp_tools(session)
                print(f"Loaded tools: {[t.name for t in tools]}")

                agent = create_react_agent(model, tools)
                agent_with_recursion_limit = agent.with_config(recursion_limit=12)
                print("Agent created, starting dialogue...")

                begin_index = 0
                for index, row in tqdm(
                    data_tmp.iterrows(),
                    total=data_tmp.shape[0]
                ):
                    if row["code"] == 0 or index < begin_index:
                        continue

                    print(f"========ROW {agent_name}{index}========")
                    q = get_mcp_prompt(row)

                    messages = [
                        SystemMessage(content=system),
                        HumanMessage(content=q)
                    ]

                    # Convert to standard format
                    raw_messages = []
                    for m in messages:
                        if isinstance(m, SystemMessage):
                            raw_messages.append(
                                {"role": "system", "content": m.content}
                            )
                        elif isinstance(m, HumanMessage):
                            raw_messages.append(
                                {"role": "user", "content": m.content}
                            )
                        elif isinstance(m, AIMessage):
                            raw_messages.append(
                                {"role": "assistant", "content": m.content}
                            )

                    try:
                        print("⏳ AI is thinking...")

                        final_response_content = ""
                        async for chunk in agent_with_recursion_limit.astream(
                            {"messages": messages},
                            stream_mode="values"
                        ):
                            last_msg = chunk["messages"][-1]

                            # A. Print tool calls requested by the AI
                            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                                for tool_call in last_msg.tool_calls:
                                    print(
                                        f"👉 [AI Request] Calling tool: {tool_call['name']}"
                                    )
                                    print(
                                        f"   Arguments: {tool_call['args']}"
                                    )

                            # B. Print tool return values (this was previously missing)
                            elif isinstance(last_msg, ToolMessage):
                                print(
                                    f"🛠️ [Tool Return] ID: {last_msg.tool_call_id}"
                                )
                                content_show = str(last_msg.content)
                                if len(content_show) > 200:
                                    content_show = (
                                        content_show[:200]
                                        + "...(content truncated)"
                                    )
                                print(f"   Result: {content_show}")

                            # C. Record final answer
                            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                                final_response_content = last_msg.content

                        # === End of streaming ===
                        print(f"\n🤖 [AI Answer]:\n{final_response_content}")
                        data_tmp.at[
                            index,
                            f'mcp_response_{agent_name}'
                        ] = final_response_content

                    except Exception as e:
                        print(
                            f"An error occurred while processing question {index + 1}: {str(e)}"
                        )
                        data_tmp.at[
                            index,
                            f'mcp_response_{agent_name}'
                        ] = f"Error: {str(e)}"
                        final_response_content = f"Error: {str(e)}"

                    if index % 3 == 0:
                        data_tmp.to_json(
                            output_file2,
                            force_ascii=False,
                            orient="records",
                            indent=2
                        )

                    await asyncio.sleep(0.5)

                data_tmp.to_json(
                    output_file2,
                    force_ascii=False,
                    orient="records",
                    indent=2
                )

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


# 5. Ensure the main entry point is correct
if __name__ == "__main__":
    print("Starting main program...")
    asyncio.run(main())

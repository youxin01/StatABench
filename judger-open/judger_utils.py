import base64
import os
from typing import Any, Dict, List, Union


PaperInput = Dict[str, str]


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def encode_pdf_as_data_url(file_path: str) -> str:
    with open(file_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


def normalize_paper_type(file_path: str, paper_type: str = "auto") -> str:
    normalized_type = (paper_type or "auto").lower()
    if normalized_type == "auto":
        ext = os.path.splitext(file_path)[1].lower()
        return "pdf" if ext == ".pdf" else "text"
    if normalized_type in {"md", "markdown", "txt"}:
        return "text"
    if normalized_type not in {"text", "pdf"}:
        raise ValueError(f"Unsupported paper type: {paper_type}")
    return normalized_type


def load_paper_input(file_path: str, paper_type: str = "auto") -> PaperInput:
    if not file_path:
        raise ValueError("paper_path is required.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Paper file not found: {file_path}")

    normalized_type = normalize_paper_type(file_path, paper_type)
    if normalized_type == "pdf":
        return {
            "type": "pdf",
            "filename": os.path.basename(file_path),
            "file_data": encode_pdf_as_data_url(file_path),
        }

    return {
        "type": "text",
        "text": read_text_file(file_path),
    }


def get_paper_reference_text(paper_input: PaperInput) -> str:
    if paper_input["type"] == "pdf":
        return (
            "The submitted paper is attached as a PDF file in this request. "
            "Please read the attached PDF directly and use it as the primary basis for evaluation."
        )
    return paper_input["text"]


def build_user_content(user_prompt: str, paper_input: PaperInput) -> Union[str, List[Dict[str, str]]]:
    if paper_input["type"] == "pdf":
        return [
            {
                "type": "input_file",
                "filename": paper_input["filename"],
                "file_data": paper_input["file_data"],
            },
            {
                "type": "input_text",
                "text": user_prompt,
            },
        ]
    return user_prompt


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    output_items = getattr(response, "output", None) or []
    collected_text = []
    for item in output_items:
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if isinstance(text, str):
                collected_text.append(text)
                continue
            value = getattr(text, "value", None)
            if value:
                collected_text.append(value)

    if collected_text:
        return "\n".join(collected_text)

    raise ValueError("Unable to extract text content from model response.")


def request_judger_completion(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    paper_input: PaperInput,
) -> str:
    if paper_input["type"] == "pdf":
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=[
                {
                    "role": "user",
                    "content": build_user_content(user_prompt, paper_input),
                }
            ],
        )
        return extract_response_text(response)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        n=1,
    )
    return response.choices[0].message.content

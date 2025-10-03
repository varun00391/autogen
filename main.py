from typing import Any, Dict, List
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from autogen_agentchat.messages import UserMessage
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

from dotenv import load_dotenv
import os
import fitz # PyMuPDF

# source .venv/bin/activate 

load_dotenv()

# ---------------- Tools ----------------
def pdf_tool1(pdf_path: str) -> str:
    """Extracts text from PDF1."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return f"PDF1 content:\n{text}"

def pdf_tool2(pdf_path: str) -> str:
    """Extracts text from PDF2."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return f"PDF2 content:\n{text}"

# Create a Groq-compatible model client
model_client = OpenAIChatCompletionClient(
    model=os.environ.get("OPENAI_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct"), # meta-llama/llama-4-scout-17b-16e-instruct, openai/gpt-oss-20b
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.groq.com/openai/v1"),
    api_key=os.environ["GROQ_API_KEY"],
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
)

# ---------------- Agents ----------------
first_agent = AssistantAgent(
    name="first_agent",
    model_client=model_client,
    handoffs=["planner_agent"],
    tools=[pdf_tool1],
    system_message="""You are the first agent. Extract text from PDF1 using pdf_tool1, 
    then hand off the result to the planner agent."""
)

second_agent = AssistantAgent(
    name="second_agent",
    model_client=model_client,
    handoffs=["planner_agent"],
    tools=[pdf_tool2],
    system_message="""You are the second agent. Extract text from PDF2 using pdf_tool2, 
    then hand off the result to the planner agent."""
)

planner_agent = AssistantAgent(
    name="planner_agent",
    model_client=model_client,
    handoffs=[],  # terminate after comparison
    system_message="""
You are the planner agent. 

You will receive extracted text from first_agent and second_agent (two invoices). 
Your task is to **compare the invoices and extract the following structured information** from both:

- date
- customer_name
- item
- quantity
- price
- total

Return the results in **JSON format** exactly like this:

{
    "invoice1": {
        "date": "...",
        "customer_name": "...",
        "item": "...",
        "quantity": "...",
        "price": "...",
        "total": "..."
    },
    "invoice2": {
        "date": "...",
        "customer_name": "...",
        "item": "...",
        "quantity": "...",
        "price": "...",
        "total": "..."
    },
    "differences": "Describe any differences between invoice1 and invoice2 here."
}

If any field is missing in the invoice, use null. 
After producing this JSON, **TERMINATE**.
"""
)

termination =  TextMentionTermination("TERMINATE")  #HandoffTermination(target="user") |
# termination = MaxMessageTermination(max_messages=20)
team = Swarm([first_agent, second_agent,planner_agent], termination_condition=termination)

# ---------------- Task ----------------
task = """
pdf1_path: Invoice1.pdf
pdf2_path: Invoice2.pdf
"""

async def run_team_stream() -> None:
    # Run the task once, letting the agents handle everything
    await Console(team.run_stream(task=task))


async def main():
    try:
        await run_team_stream()
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())





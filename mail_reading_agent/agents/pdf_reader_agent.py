import os, asyncio

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination

from tools.pdf_reader_tool import pdf_reader_tool
from agents.file_intake_agent import file_agent

load_dotenv()

# ---------------- Model Client ----------------
model_client = OpenAIChatCompletionClient(
    model=os.environ.get("OPENAI_MODEL_NAME", "openai/gpt-oss-120b"),
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

# ---------------- Agent ----------------
pdf_reader_agent = AssistantAgent(
    name="pdf_reader_agent",
    model_client=model_client,
    handoffs=["file_agent"],
    tools=[pdf_reader_tool],
    system_message="""
You are a PDF Reader assistant.
You receive PDF files from the File Agent.
Use the pdf_reader_tool to extract text from them.
Once reading is done, tell the file_intake_agent to check for the next file.
If there are no files left, say 'TERMINATE'.
"""
)


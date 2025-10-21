import os
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

from tools.file_intake_tool import file_intake_tool  # Import the tool
# from pdf_reader_agent import pdf_reader_agent  # Import the PDF reader agent

load_dotenv()

# --- Model Client ---
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

# --- Assistant Agent ---
file_agent = AssistantAgent(
    name="file_agent",
    model_client=model_client,
    handoffs=["pdf_reader_agent"],
    tools=[file_intake_tool],
    system_message="""
You are a File Intake assistant.
Your job:
1. Check for new PDF or Excel files in the attachments folder using file_intake_tool.
2. If you find a new file, instruct the pdf_reader_agent to read it.
3. If no new files are found, respond with 'TERMINATE'.
"""
)


# # --- Main Execution ---
# async def main():
#     result = await file_intake_agent.run(task="Monitor PDF and Excel files in attachments folder.")
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(main())

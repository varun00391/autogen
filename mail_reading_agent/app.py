import os, asyncio
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import Swarm

from agents.pdf_reader_agent import pdf_reader_agent
from agents.file_intake_agent import file_agent

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

termination =  TextMentionTermination("TERMINATE")  #HandoffTermination(target="user") |
# termination = MaxMessageTermination(max_messages=20)
team = Swarm([file_agent, pdf_reader_agent], termination_condition=termination)

# from autogen_agentchat.messages import UserMessage

task = """Idenify the PDF files from attachment folder and then 
read the PDF files from attachments folder one by one"""


async def run_team_stream() -> None:
    # Run the task once, letting the agents handle everything
    await Console(team.run_stream(task=task))


import asyncio

async def main():
    try:
        await run_team_stream()
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
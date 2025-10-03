import streamlit as st
import asyncio
import tempfile
import fitz  # PyMuPDF
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------- Tools ----------------
def pdf_tool1(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
    return f"PDF1 content:\n{text}"

def pdf_tool2(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
    return f"PDF2 content:\n{text}"

# ---------------- Model client ----------------
model_client = OpenAIChatCompletionClient(
    model=os.environ.get("OPENAI_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct"),
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
    system_message="""You are the first agent. Extract text from PDF1 using pdf_tool1, then hand off the result to the planner agent."""
)

second_agent = AssistantAgent(
    name="second_agent",
    model_client=model_client,
    handoffs=["planner_agent"],
    tools=[pdf_tool2],
    system_message="""You are the second agent. Extract text from PDF2 using pdf_tool2, then hand off the result to the planner agent."""
)

planner_agent = AssistantAgent(
    name="planner_agent",
    model_client=model_client,
    handoffs=[],  # terminate after comparison
    system_message="""
You are the planner agent. 

You will receive extracted text from first_agent and second_agent (two invoices). 
Your task is to compare the invoices and extract the following structured information from both:

- date
- customer_name
- item
- quantity
- price
- total

Return the results in JSON format exactly like this:

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

If any field is missing, use null. After producing this JSON, TERMINATE.
"""
)

termination = TextMentionTermination("TERMINATE")

# ---------------- Streamlit UI ----------------
st.title("Invoice Comparison System")

uploaded_pdf1 = st.file_uploader("Upload Invoice 1", type=["pdf"])
uploaded_pdf2 = st.file_uploader("Upload Invoice 2", type=["pdf"])

if st.button("Compare Invoices"):
    if not uploaded_pdf1 or not uploaded_pdf2:
        st.warning("Please upload both PDFs.")
    else:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
            tmp1.write(uploaded_pdf1.read())
            tmp2.write(uploaded_pdf2.read())
            tmp1_path = tmp1.name
            tmp2_path = tmp2.name

        # Define the task dynamically
        task = f"""
        pdf1_path: {tmp1_path}
        pdf2_path: {tmp2_path}
        """

        team = Swarm([first_agent, second_agent, planner_agent], termination_condition=termination)

        async def run_task():
            task_result = await Console(team.run_stream(task=task))
            # Collect messages and flatten lists
            messages = []
            for m in task_result.messages:
                if hasattr(m, "content"):
                    if isinstance(m.content, list):
                        messages.append("\n".join(str(item) for item in m.content))
                    else:
                        messages.append(str(m.content))
            return "\n\n".join(messages)

        result = asyncio.run(run_task())
        st.text_area("Comparison Result (JSON)", result, height=400)

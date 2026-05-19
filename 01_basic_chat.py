# ============================================================
# 01_basic_chat.py — Simplest possible LangChain chat example
# ============================================================
#
# All dependencies are in requirements.txt. Install with:
#   pip install -r requirements.txt
#
# API keys are loaded from the .env file in this folder.
#
# ============================================================

# --- 1. Load environment variables from a .env file ---------
from dotenv import load_dotenv  # helper to read .env files
load_dotenv(override=True)      # populates os.environ with .env values
                                # override=True → .env wins over existing shell vars

# --- 2. Import message types --------------------------------
# LangChain wraps every message in a typed object so the model
# knows *who* is speaking:
#   SystemMessage  → sets the model's persona / instructions
#   HumanMessage   → the user's input
#   AIMessage      → the model's reply (returned by .invoke())
from langchain_core.messages import SystemMessage, HumanMessage

# --- 3. Pick ONE chat model (uncomment the one you want) ----

# ── Option A: Anthropic (Claude) ────────────────────────────
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(
    model="claude-sonnet-4-20250514",  # model id string
    temperature=0.7,                   # 0 = deterministic, 1 = creative
    max_tokens=1024,                   # cap on response length
)

# ── Option B: OpenAI (GPT) ──────────────────────────────────
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(
#     model="gpt-4o",       # or "gpt-4o-mini" for cheaper/faster
#     temperature=0.7,      # same 0-1 creativity knob
#     max_tokens=1024,      # reads OPENAI_API_KEY from .env automatically
# )

# ── Option C: xAI (Grok) ────────────────────────────────────
# from langchain_xai import ChatXAI
# model = ChatXAI(
#     model="grok-3-latest",    # Grok model id
#     temperature=0.7,
#     max_tokens=1024,
# )

# ── Option D: HuggingFace (serverless Inference API) ────────
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",  # any text-gen model on HF
#     temperature=0.7,
#     max_new_tokens=1024,
# )
# model = ChatHuggingFace(llm=llm)             # wraps it in the chat interface

# ── Option E: Groq (fast inference for open-source models) ──
# from langchain_groq import ChatGroq
# model = ChatGroq(
#     model="llama-3.3-70b-versatile",  # runs Llama/Mixtral on Groq hardware
#     temperature=0.7,
#     max_tokens=1024,
# )

# --- 4. Build the conversation as a list of messages ---------
messages = [
    # The system message is invisible to the user — it tells the
    # model *how* to behave for the entire conversation.
    SystemMessage(content="You are a friendly assistant who explains things simply."),

    # The human message is the actual question / prompt from the user.
    HumanMessage(content="In 3 sentences, what is LangChain and why should I learn it?"),
]

# --- 5. Send the messages to the model -----------------------
# .invoke() is the standard synchronous call in LangChain.
# It sends the full list of messages and returns a single
# AIMessage object containing the model's response.
response = model.invoke(messages)

# --- 6. Print the result -------------------------------------
# response         → full AIMessage (includes metadata, token usage, etc.)
# response.content → just the text string the model generated
print("Full AIMessage object:")
print(response)
print("\n--- Model's text reply ---")
print(response.content)

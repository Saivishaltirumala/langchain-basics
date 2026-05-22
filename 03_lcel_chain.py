# ============================================================
# 03_lcel_chain.py — Multi-step workflow using LCEL pipes
# ============================================================
#
# LCEL (LangChain Expression Language) lets you connect
# components with the | operator, just like Unix pipes:
#
#   Unix:      cat file.txt | grep "error" | sort
#   LangChain: prompt | model | parser
#
# Each component's output becomes the next component's input.
#
# This script builds a 2-step pipeline:
#   Step 1: Generate a joke about a topic
#   Step 2: Translate that joke into Spanish
#
# Dependencies: pip install -r requirements.txt
# ============================================================

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser extracts just the text string from an AIMessage.
# Without it, you get the full AIMessage object (with metadata).
# With it, you get a clean string — ready to pipe into the next step.
#
#   Without: AIMessage(content="Why did the...", metadata={...})
#   With:    "Why did the..."
from langchain_core.output_parsers import StrOutputParser


# --- 1. Create the model and parser -------------------------

model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.7,
    max_tokens=1024,
)

parser = StrOutputParser()


# --- 2. Build Chain 1: Topic → Joke -------------------------
#
# This prompt takes a {topic} and asks the model to write a joke.

joke_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a stand-up comedian. Keep jokes short — 2-3 lines max."),
    ("human", "Tell me a joke about {topic}"),
])

# The | operator connects: prompt → model → parser
#
# Data flows left to right:
#   joke_prompt receives {"topic": "python"}
#       → fills template → "Tell me a joke about python"
#           → model generates → AIMessage("Why did the...")
#               → parser extracts → "Why did the..."
joke_chain = joke_prompt | model | parser


# --- 3. Build Chain 2: English text → Spanish translation ----
#
# This prompt takes a {text} and translates it.

translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Translate the given text to Spanish. Only output the translation, nothing else."),
    ("human", "{text}"),
])

translate_chain = translate_prompt | model | parser


# --- 4. Connect both chains into one pipeline ----------------
#
# Problem: joke_chain outputs a string like "Why did the..."
#          translate_chain expects a dict like {"text": "Why did the..."}
#
# We need a bridge to convert:  string → {"text": string}
#
# RunnableLambda wraps any Python function into a LangChain
# component that can be used in a | pipeline.

from langchain_core.runnables import RunnableLambda

# This function takes the joke string and wraps it in the dict
# format that translate_prompt expects.
format_for_translation = RunnableLambda(lambda joke: {"text": joke})

# Now we can pipe everything together:
#
#   {"topic": "python"}
#       → joke_prompt     fills template with topic
#       → model           generates the joke
#       → parser          extracts clean string
#       → lambda          wraps string into {"text": "..."}
#       → translate_prompt fills template with text
#       → model           generates translation
#       → parser          extracts clean string
#       → final result: Spanish joke as a string

full_chain = joke_chain | format_for_translation | translate_chain


# --- 5. Visualize the pipeline ------------------------------

print("=" * 55)
print("PIPELINE STRUCTURE")
print("=" * 55)
print("""
  {"topic": "python"}
       │
       ▼
  ┌─────────────┐
  │ joke_prompt  │  "Tell me a joke about python"
  └──────┬──────┘
         │  (pipe)
         ▼
  ┌─────────────┐
  │    model     │  AIMessage("Why did the python...")
  └──────┬──────┘
         │  (pipe)
         ▼
  ┌─────────────┐
  │   parser     │  "Why did the python..."
  └──────┬──────┘
         │  (pipe)
         ▼
  ┌─────────────┐
  │   lambda     │  {"text": "Why did the python..."}
  └──────┬──────┘
         │  (pipe)
         ▼
  ┌──────────────────┐
  │ translate_prompt  │  "Translate: Why did the python..."
  └───────┬──────────┘
          │  (pipe)
          ▼
  ┌─────────────┐
  │    model     │  AIMessage("¿Por qué la python...")
  └──────┬──────┘
         │  (pipe)
         ▼
  ┌─────────────┐
  │   parser     │  "¿Por qué la python..."
  └──────┬──────┘
         │
         ▼
    Final output (string)
""")


# --- 6. Run the full pipeline --------------------------------

topic = "programming"

print("=" * 55)
print(f"TOPIC: {topic}")
print("=" * 55)

# Run chain 1 alone first to see the intermediate joke
print("\nStep 1 — English joke:")
joke = joke_chain.invoke({"topic": topic})
print(f"  {joke}")

# Run chain 2 alone to see translation in action
print("\nStep 2 — Spanish translation:")
translated = translate_chain.invoke({"text": joke})
print(f"  {translated}")

# Now run the FULL pipeline in one call — same result, one line
print("\n" + "=" * 55)
print("FULL PIPELINE (single .invoke() call):")
print("=" * 55)
result = full_chain.invoke({"topic": topic})
print(f"\n  {result}")

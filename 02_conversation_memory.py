# ============================================================
# 02_conversation_memory.py — Adding memory to a chat model
# ============================================================
#
# Without memory, every .invoke() call is a fresh conversation.
# The model has NO idea what you said 10 seconds ago.
#
# This script solves that using RunnableWithMessageHistory,
# which automatically saves and injects past messages into
# every new call.
#
# Dependencies: pip install -r requirements.txt
# ============================================================

from dotenv import load_dotenv
load_dotenv(override=True)

# --- 1. Import the building blocks ---------------------------

from langchain_anthropic import ChatAnthropic

# ChatPromptTemplate   → defines the structure of every request
# MessagesPlaceholder  → a slot that gets filled with past chat history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# InMemoryChatMessageHistory → stores messages in a Python dict (RAM)
# Lost when the script exits. For production, you'd use Redis/SQL/etc.
from langchain_core.chat_history import InMemoryChatMessageHistory

# RunnableWithMessageHistory → glue that connects the chain to the store
from langchain_core.runnables.history import RunnableWithMessageHistory


# --- 2. Create the chat model --------------------------------

model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.7,
    max_tokens=1024,
)


# --- 3. Build the prompt template ----------------------------
#
# This is the structure of EVERY message sent to the model.
# Think of it as a form with blank slots that get filled in
# before each call.

prompt = ChatPromptTemplate.from_messages([
    # Slot 1: system instruction (fixed)
    ("system", "You are a friendly assistant with a good memory."),

    # Slot 2: past conversation history (filled automatically)
    # "chat_history" is just a variable name — RunnableWithMessageHistory
    # will inject all previous messages here before each call.
    MessagesPlaceholder(variable_name="chat_history"),

    # Slot 3: the new human message (filled by us at invoke time)
    ("human", "{input}"),
])


# --- 4. Create the chain using LCEL (pipe operator) ----------
#
# prompt | model
#
# This means: take the prompt, fill in the blanks,
# then pass the result to the model.
# The | operator is LangChain Expression Language (LCEL) —
# it connects components left-to-right like a Unix pipe.

chain = prompt | model


# --- 5. Set up the session history store ----------------------
#
# This dict holds one InMemoryChatMessageHistory per session_id.
# Each session_id gets its own isolated conversation history.
#
# Think of it like:
#   store = {
#       "user_123": [HumanMessage(...), AIMessage(...), ...],
#       "user_456": [HumanMessage(...), AIMessage(...), ...],
#   }

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Called automatically before every .invoke().
    Returns the history object for this session.
    If this session_id is new, creates an empty history.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# --- 6. Wrap the chain with memory ---------------------------
#
# RunnableWithMessageHistory does 3 things automatically:
#   BEFORE each call → loads past messages into "chat_history" placeholder
#   SENDS to model   → model sees full conversation context
#   AFTER each call  → saves the new human + AI messages to the store

chain_with_memory = RunnableWithMessageHistory(
    chain,                                      # the chain to wrap
    get_session_history,                         # function that returns history
    input_messages_key="input",                  # which key in our input holds the new message
    history_messages_key="chat_history",         # which placeholder to inject history into
)


# --- 7. Simulate a 2-turn conversation -----------------------
#
# The config dict tells RunnableWithMessageHistory WHICH session
# to load/save history for. Different session_ids = different
# conversations (like different users or browser tabs).

config = {"configurable": {"session_id": "demo_session_1"}}

# Turn 1: Tell the model our name
print("=" * 50)
print("TURN 1: Telling the model our name")
print("=" * 50)
response1 = chain_with_memory.invoke(
    {"input": "Hi there! My name is Sai."},     # new message
    config=config,                                # which session
)
print(f"Human: Hi there! My name is Sai.")
print(f"AI:    {response1.content}")


# Turn 2: Ask the model to recall our name
# We NEVER mention the name again — the model can only know it
# if the memory system successfully injected Turn 1's messages.
print()
print("=" * 50)
print("TURN 2: Asking the model to recall our name")
print("=" * 50)
response2 = chain_with_memory.invoke(
    {"input": "What's my name? Do you remember?"},
    config=config,                                 # same session → same history
)
print(f"Human: What's my name? Do you remember?")
print(f"AI:    {response2.content}")


# --- 8. Show what's actually stored in memory ----------------
#
# Let's peek inside the store to see what was saved.

print()
print("=" * 50)
print("WHAT'S INSIDE THE MEMORY STORE")
print("=" * 50)

history = store["demo_session_1"]
for i, message in enumerate(history.messages):
    role = message.__class__.__name__       # HumanMessage or AIMessage
    content = message.content[:80]          # truncate for readability
    print(f"  [{i}] {role}: {content}...")


# --- 9. Prove that a NEW session has NO memory ---------------

print()
print("=" * 50)
print("TURN 3: Different session — no memory")
print("=" * 50)

different_config = {"configurable": {"session_id": "demo_session_2"}}
response3 = chain_with_memory.invoke(
    {"input": "What's my name?"},
    config=different_config,                       # different session → empty history
)
print(f"Human: What's my name?")
print(f"AI:    {response3.content}")

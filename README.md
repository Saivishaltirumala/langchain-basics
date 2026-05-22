# LangChain Basics

A step-by-step learning series for LangChain fundamentals. Each script is standalone and heavily commented.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
```

## Scripts

### 01 — Basic Chat (`01_basic_chat.py`)

The simplest possible LangChain interaction. Sends a `SystemMessage` + `HumanMessage` to a chat model and prints the response.

**Concepts covered:**
- `SystemMessage`, `HumanMessage`, `AIMessage` — LangChain's message types
- `.invoke()` — synchronous call to the model
- `temperature` and `max_tokens` — model parameters
- `load_dotenv()` — loading API keys from `.env`
- 5 provider options: Anthropic, OpenAI, xAI (Grok), HuggingFace, Groq

```bash
python 01_basic_chat.py
```

### 02 — Conversation Memory (`02_conversation_memory.py`)

Adds memory to a chat model so it remembers previous messages across turns. The user tells the AI their name, then asks the AI to recall it.

**Concepts covered:**
- `ChatPromptTemplate` + `MessagesPlaceholder` — reusable prompt with slots
- LCEL pipe operator (`prompt | model`) — chaining components
- `InMemoryChatMessageHistory` — in-RAM message storage
- `RunnableWithMessageHistory` — automatic history load/inject/save
- Session isolation — different `session_id` = different conversation

```bash
python 02_conversation_memory.py
```

### 03 — LCEL Chains (`03_lcel_chain.py`)

Multi-step workflow using the LCEL pipe (`|`) operator. Generates a joke about a topic, then translates it to Spanish — all in one pipeline.

**Concepts covered:**
- LCEL `|` operator — connecting components left-to-right like Unix pipes
- `StrOutputParser` — extracts clean text from `AIMessage` objects
- `RunnableLambda` — wraps any Python function into a pipeable component
- Chaining two separate prompt→model→parser sequences into one pipeline
- Running sub-chains independently vs the full pipeline in a single `.invoke()`

```bash
python 03_lcel_chain.py
```

### 04 — MCP Agent (`04_mcp_agent.py`)

Connects a LangChain agent to an external MCP (Model Context Protocol) weather server. The agent discovers tools at runtime, decides which to call, and answers weather questions using real forecast data.

**Concepts covered:**
- `MultiServerMCPClient` — connects to MCP servers and discovers tools dynamically
- `create_agent` — builds a ReAct-style tool-calling loop (Reason → Act → Observe)
- MCP stdio transport — communicating with a server via subprocess stdin/stdout
- Full agent trace — seeing the LLM's reasoning, tool calls, and final answer

**Requires:** Node.js (`npx`) for the weather MCP server.

```bash
python 04_mcp_agent.py
```

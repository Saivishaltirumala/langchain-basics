# ============================================================
# 04_mcp_agent.py — LangChain Agent + MCP Server
# ============================================================
#
# MCP (Model Context Protocol) is a standard for connecting
# AI models to external tools/services. Instead of writing
# tool functions yourself, an MCP server exposes them and
# LangChain discovers them automatically.
#
# This script connects to a weather MCP server, discovers its
# tools at runtime, and lets an LLM call them to answer
# weather questions.
#
# Extra dependencies (beyond requirements.txt):
#   pip install langchain langchain-mcp-adapters langchain-openai
#
# Also requires Node.js (npx) for the weather MCP server.
#
# ============================================================

import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)


# --- 1. Import the building blocks ---------------------------

# MultiServerMCPClient — connects to one or more MCP servers
# and discovers their tools automatically.
from langchain_mcp_adapters.client import MultiServerMCPClient

# create_agent — builds a tool-calling agent loop:
#   LLM decides → call tool → get result → LLM decides again → ... → final answer
# This replaced the older AgentExecutor in LangChain 1.3+
from langchain.agents import create_agent

# Chat model — Anthropic Claude for tool-calling
from langchain_anthropic import ChatAnthropic


# --- 2. Define the async main function ----------------------
#
# MCP communication is async (uses stdio/network streams),
# so everything must run inside an async function.

async def main():

    # --- 3. Configure the MCP client -------------------------
    #
    # MultiServerMCPClient can connect to multiple MCP servers
    # at once. Each server gets a name and connection config.
    #
    # Here we connect to ONE server: a weather server that
    # provides US weather forecasts and alerts via the
    # National Weather Service API (free, no key needed).
    #
    # "command" + "args" = what to run in a subprocess
    # "transport": "stdio" = communicate via stdin/stdout pipes
    #
    # When get_tools() is called, the client:
    #   1. Spawns: npx -y @h1deya/mcp-server-weather
    #   2. Sends a "list tools" request over stdio
    #   3. Server responds with its available tools
    #   4. Client wraps each one as a LangChain Tool object

    client = MultiServerMCPClient(
        {
            "weather": {
                "command": "npx",
                "args": ["-y", "@h1deya/mcp-server-weather"],
                "transport": "stdio",
            },
        }
    )

    # --- 4. Discover tools from the MCP server ----------------
    #
    # This is the key MCP benefit — tools are NOT hardcoded.
    # The server tells us what it can do at runtime.

    tools = await client.get_tools()

    print("=" * 55)
    print("TOOLS DISCOVERED FROM MCP SERVER")
    print("=" * 55)
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}")
    print()

    # --- 5. Create the chat model -----------------------------

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,             # 0 for factual tool-calling tasks
        max_tokens=1024,
    )

    # --- 6. Create the agent ----------------------------------
    #
    # create_agent() builds a loop (internally uses LangGraph):
    #
    #   ┌─────────────────────────────────────────────────┐
    #   │                                                 │
    #   │  User: "What's the weather in New York?"        │
    #   │       │                                         │
    #   │       ▼                                         │
    #   │  ┌─────────┐                                    │
    #   │  │   LLM    │ "I should call get_forecast       │
    #   │  │ (thinks) │  with lat=40.7, lon=-74.0"        │
    #   │  └────┬─────┘                                   │
    #   │       │ tool call                               │
    #   │       ▼                                         │
    #   │  ┌──────────────┐                               │
    #   │  │  MCP Server   │ runs get_forecast()          │
    #   │  │  (weather)    │ returns forecast JSON         │
    #   │  └──────┬───────┘                               │
    #   │         │ tool result                           │
    #   │         ▼                                       │
    #   │  ┌─────────┐                                    │
    #   │  │   LLM    │ "The weather in NYC is 72°F..."   │
    #   │  │ (final)  │  → returns to user                │
    #   │  └─────────┘                                    │
    #   │                                                 │
    #   │  If LLM needed MORE info, it would loop back    │
    #   │  and call another tool. The loop continues       │
    #   │  until the LLM decides it has enough info.      │
    #   └─────────────────────────────────────────────────┘
    #
    # This is the "ReAct" pattern:
    #   Reason → Act → Observe → Reason → ... → Answer

    agent = create_agent(
        llm,                           # the brain — decides what tools to call
        tools,                         # the discovered MCP tools
        system_prompt=(                # instructions for the agent
            "You are a helpful weather assistant. "
            "Use the available tools to answer weather questions. "
            "Always provide clear, concise answers."
        ),
    )

    # --- 7. Run the agent -------------------------------------
    #
    # ainvoke() is the async version of invoke().
    # The agent will:
    #   1. Read the user's question
    #   2. Decide which tool(s) to call
    #   3. Call the MCP weather server
    #   4. Read the result
    #   5. Formulate a human-readable answer

    print("=" * 55)
    print("ASKING: What's the weather forecast for New York?")
    print("=" * 55)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the current weather forecast for New York City?"}]},
    )

    # --- 8. Print the result ----------------------------------
    #
    # The result contains the full message history including
    # tool calls and responses. The final message is the answer.

    # Extract the final AI message from the conversation
    final_message = result["messages"][-1]
    print(f"\nAgent's answer:\n{final_message.content}")

    # --- 9. Show the full tool-calling trace ------------------
    #
    # Let's peek at ALL messages to see the agent's reasoning.

    print()
    print("=" * 55)
    print("FULL AGENT TRACE (all messages)")
    print("=" * 55)
    for msg in result["messages"]:
        role = msg.__class__.__name__
        content = str(msg.content)[:100] if msg.content else "(empty)"
        print(f"\n  [{role}]")
        print(f"  {content}")

        # Show tool calls if the LLM decided to call a tool
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    → Tool call: {tc['name']}({tc['args']})")


# --- 10. Run the async main function -------------------------
#
# asyncio.run() creates an event loop and runs our async function.
# This is required because MCP communication (stdio pipes,
# network streams) is inherently asynchronous.

if __name__ == "__main__":
    asyncio.run(main())

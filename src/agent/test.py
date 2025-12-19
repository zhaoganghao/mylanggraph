from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.messages import AIMessage
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model

from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = init_chat_model(
    model="gpt-4o-mini",
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-l51iHp8j5cTjLZKP5tDHFg05IU7vJqHPvRh9bQinZCHi5qYi"
)
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from langchain.messages import HumanMessage


agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=InMemorySaver(),
middleware=[
        FilesystemFileSearchMiddleware(
            root_path="./",
            use_ripgrep=True,
            max_file_size_mb=10,
        ),
    ],
)
config = {
    "configurable": {
        "thread_id": "1"
    }
}
# Agent can now use glob_search and grep_search tools
result = agent.invoke({
    "messages": [HumanMessage("你好，我是小猫'")]
},config=config)
result = agent.invoke({
    "messages": [HumanMessage("你好我叫什么名字")]
},config=config)
for e in result["messages"]:
    e.pretty_print()


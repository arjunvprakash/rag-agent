
import sqlite3
from typing import Annotated, Literal, TypedDict

from langchain.messages import AIMessage, AnyMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import rag


## LLM
LLM_MODEL="llama3.2:1b"
BASE_URL="http://127.0.0.1:11434"

retriever = rag.load_retriever(False)

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL)


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage],add_messages]

def llm_node(state:ChatState):
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    print("\n\n###")
    print(f"context: {context}")
    print("###\n\n")

    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say you could not find it in the PDF.

Context:
{context}

Question:
{question}
"""
    response = llm.invoke(prompt).content
    print(f"Answer: {response}")
    return {"messages": AIMessage(content=response)}

def should_continue(state:ChatState)->Literal["llm_node","__end__"]:
    if state["messages"][-1] and state["messages"][-1].content.lower() in ["exit", "bye", "quit", "stop"]:
        return END
    return "llm_node"

def user_node(state:ChatState):
    user_message=input("Question: ")
    return {"messages": HumanMessage(content=user_message.strip())}

graph_builder = StateGraph(ChatState)

graph_builder.add_node(user_node)
graph_builder.add_node(should_continue)
graph_builder.add_node(llm_node)

graph_builder.add_edge(START,"user_node")
graph_builder.add_conditional_edges("user_node", should_continue, {"llm_node", END})
graph_builder.add_edge("llm_node","user_node")

# ims = InMemorySaver()
conn = sqlite3.connect("checkpoints.sqlite")
checkpoint = SqliteSaver(conn)
graph = graph_builder.compile(checkpointer=checkpoint)
with open("agent.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

graph.invoke({"messages":""},config={"configurable":{"thread_id":"thread-1"}})
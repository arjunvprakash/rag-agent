
import sqlite3
from typing import Annotated, Literal, TypedDict

from langchain.messages import AIMessage, AnyMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import argparse
# Internal dependencies
import rag


## LLM
LLM_MODEL="llama3.2:1b"
BASE_URL="http://127.0.0.1:11434"

reindex=False

parser = argparse.ArgumentParser(description="""
A simple local RAG project built with Ollama, LangChain, and ChromaDB.  
It lets you ask questions about a PDF using a fully local LLM setup, with embeddings stored on disk so you don’t have to re-index every time.
""")
parser.set_defaults(func=lambda x: parser.print_usage())
parser.add_argument("-r", "--reindex", action='store_true', help="force reindex of the vectorstore")
parser.add_argument("-d", "--debug", action='store_true', help="execute in debug mode")
args = parser.parse_args()
if not vars(args):
    parser.print_help()
    parser.exit(1)

debug=args.debug
retriever = rag.load_retriever(args.reindex)
llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL)

class ChatState(TypedDict):
    """LangGraph state object for Chat workflow"""
    messages: Annotated[list[AnyMessage],add_messages]
    debug: bool

def llm_node(state:ChatState):
    """LLM node for the RAG agent workflow. Fetch docs from RAG and uses LLM to generate the response for the user."""
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    if state["debug"]:
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
    print(f"Agent: {response}")
    return {"messages": AIMessage(content=response)}

def should_continue(state:ChatState)->Literal["llm_node","__end__"]:
    if state["messages"][-1] and state["messages"][-1].content.lower() in ["exit", "bye", "quit", "stop"]:
        print("Agent: Bye!")
        return END
    return "llm_node"

def user_node(state:ChatState):
    user_message=input("You: ")
    return {"messages": HumanMessage(content=user_message.strip())}



def main():
    graph_builder = StateGraph(ChatState)

    graph_builder.add_node(user_node)
    graph_builder.add_node(llm_node)

    graph_builder.add_edge(START,"user_node")
    graph_builder.add_conditional_edges("user_node", should_continue, 
            {
                "llm_node": "llm_node",
                "__end__": END,
            }
        )
    graph_builder.add_edge("llm_node","user_node")

    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    checkpoint = SqliteSaver(conn)
    graph = graph_builder.compile(checkpointer=checkpoint)
    with open("agent.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    graph.invoke({"messages":"", "debug": debug},config={"configurable":{"thread_id":"thread-1"}})


if __name__=="__main__":
    main()
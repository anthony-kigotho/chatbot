import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')  # Access your Hugging Face token
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

repo_id = "microsoft/Phi-3.5-mini-instruct"

# Set up the LLM
llm = HuggingFaceEndpoint(repo_id=repo_id, task="text-generation", temperature=0.7, huggingfacehub_api_token=HF_TOKEN)
chat_model = ChatHuggingFace(llm=llm) 

# Set up the tools
search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=10, search_depth="advanced")


def is_answer_satisfactory(evaluation_response: str) -> bool:
    # Use the model to determine if the answer is satisfactory
    return "correct" in evaluation_response.content.lower()  # Example check

def call_model(state: MessagesState):
    messages = state["messages"]
    query = messages[-1].content  # User's query

    # Fetch results from the tavily_tool based on the user's query
    search_results = tavily_tool.invoke(query)
    combined_content = " ".join(item['content'] for item in search_results)

    max_attempts = 3
    attempts = 0
    answer_response = ""

    while attempts < max_attempts:
        # Generate an answer based on the search results and the user's query using chat_model
        answer_response = chat_model.invoke(f"Answer the following query: '{query}' based on these results: {combined_content}")

        # Store the answer in messages for reasoning
        messages.append(answer_response)

        # Create a reasoning query to evaluate the answer
        evaluation_query = f"Evaluate the following answer: '{answer_response}' for the query: '{query}'. Is it correct or wrong? Why or why not?"
        evaluation_response = chat_model.invoke(evaluation_query)

        # Check if the evaluation response is satisfactory
        if is_answer_satisfactory(evaluation_response):
            break  # Exit loop if the answer is satisfactory

        attempts += 1  # Increment the attempt counter

    return {"messages": [query, answer_response]}


workflow = StateGraph(MessagesState)


workflow.add_node("agent", call_model)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Compile the graph with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Function to stream updates
def stream_graph_updates(user_input: str, config: dict):
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Run the chatbot
config = {"configurable": {"thread_id": "1"}}
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input, config)
    except Exception as e:
        print("An error occurred:", e)
        break
import os
import json
from turtledemo.sorting_animate import start_qsort
from typing import TypedDict
from dotenv import load_dotenv
from imap_tools import MailBox, AND

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

load_dotenv()

IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_FOLDER = "INBOX"

CHAT_MODEL = "qwen3:4b"

print("Welcome to Maily, your own personal smart email assistant!\n\n")

class ChatState(TypedDict):
    messages: list
    # It means the ChatState dictionary must have a key "messages", and the value for that key is expected to be a list.

def connect():
    mail_box = MailBox(IMAP_HOST)
    mail_box.login(IMAP_USER, IMAP_PASSWORD)

    return mail_box

@tool
def list_unread_emails():
    """Return a bullet list of every UNREAD email's UID, subject, date, and sender."""

    print("List unread emails tool called :)")

    with connect() as mb:
        unread = list(mb.fetch(criteria= AND(seen=False), headers_only=True, mark_seen=False))

    if not unread:
        return "No unread emails found :("

    response = json.dumps([
        {
            "uid": mail.uid,
            "subject": mail.subject,
            "date": mail.date.astimezone().strftime("%Y-%m-%d %H:%M"),
            "sender": mail.from_,
        } for mail in unread
    ])

    return response


@tool
def summarize_email(uid):
    """Summarize a single email given its IMAP UID. Return a short summary of the email's content / body in plain text."""

    print(f"Summarize email tool called on {uid}")

    with connect() as mb:
        mail = next(mb.fetch(AND(uid=uid), mark_seen=False), None) # next grabs the first email that matches the criteria of AND

        if not mail:
            return f"Could not summarize email with {uid}"

        prompt = (
            "Summarize this email concisely:\n\n"
            f"Subject: {mail.subject}\n"
            f"Sender: {mail.from_}\n"
            f"Date: {mail.date}\n\n"
            f"Content:\n{mail.text or mail.html}"
        )

        return raw_llm.invoke(prompt)


@tool
def recommend_response(uid):
    """
    Analyze the email’s content and emotional tone to craft a thoughtful response.

    The function fetches the email by its IMAP UID and generates a reply that reflects understanding of the message and feelings expressed.

    The response should include a greeting at the start and a proper email sign-off at the end, with blank lines separating the greeting, body, and sign-off for readability.

    Make sure the greeting and sign-off are made in parallel to the emotion of the email and your understanding of the email.
    """

    print("Recommend response tool called :)")


    with connect() as mb:
        mail = next(mb.fetch(AND(uid=uid), mark_seen=False), None)

        if not mail:
            return f"Could not recommend response email with {uid}"

    prompt = (
        "You are an AI assistant that reads an email and understands both the information and the emotions behind it. "
        "Based on this understanding, write a thoughtful and appropriate response given the following email\n\n"
        f"Subject: {mail.subject}\n"
        f"Sender: {mail.from_}\n"
        f"Date: {mail.date}\n\n"
        f"Content:\n{mail.text or mail.html}\n\n"
        "Please include a thoughtful greeting at the beginning and a proper email sign-off at the end. "
        "Make sure there is a blank line separating the greeting from the main response and another blank line separating the response from the sign-off."
        "Based on the information and emotions, also use an appropriate response-length and tone"
    )


llm = init_chat_model(CHAT_MODEL, model_provider="ollama")
llm = llm.bind_tools([list_unread_emails, summarize_email, recommend_response])

raw_llm = init_chat_model(CHAT_MODEL, model_provider="ollama") # making this because it is not necessary to
# use an agentic AI for the simple task of summarizing an email


# this is langraph
# state["messages"] is basically just the last messages
def llm_node(state):
    # Send the full message history to the LLM to generate a new response,
    # then return an updated state including all previous messages plus the new response.
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def router(state):
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "end"
# Syntax
# getattr(object, attribute, default)

# Parameter Values
# Parameter	Description
# object:	 Required. An object.
# attribute: The name of the attribute you want to get the value from
# default:	 Optional. The value to return if the attribute does not exist

tool_node = ToolNode([list_unread_emails, summarize_email, recommend_response])

def tools_node(state):
    result = tool_node.invoke(state)
    return {
        "messages": state["messages"] + result["messages"]
    } # returns a new dictionary with the added results from using the tools


builder = StateGraph(ChatState)
builder.add_node("llm", llm_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", router, {"tools": "tools", "end": END})
# if the router returns tools, then it will route to the tools node and if it returns end then it will end
builder.add_edge("tools", "llm")

graph = builder.compile()

if __name__ == "__main__":
    state = {"messages": []}
    # "messages" is a key in state that leads to all the messages(formatted properly). each message is its own dictionary.
    # in this format: {"role": "...", "content": "..."}
    print("Type an instruction or 'quit' to exit.")

    while True:
        user_message = input("> ")

        if user_message.lower() == "quit":
            print("Thank you for using Maily!")
            break

        state["messages"].append({"role": "user", "content": user_message}) # Add user message to the message history

        state = graph.invoke(state) # Invoke AI (which adds AI message to state["messages"] in the proper format(automatically).

        print(state["messages"][-1].content) # Print AI message because it is now the last one in state["messages"]


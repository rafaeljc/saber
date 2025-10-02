import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage


chatbot = st.session_state.get("chatbot", None)


def display_message(message: HumanMessage | AIMessage) -> None:
    """Display a message in chat.

    Args:
        message (HumanMessage | AIMessage): The message to display.
    """
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = None
    if role:
        with st.chat_message(role):
            st.markdown(message.content)
    else:
        st.error(f"Invalid message type.")


def display_chat_history() -> None:
    """Display the chat history."""
    chat_history = chatbot.get_chat_history()
    for message in chat_history:
        display_message(message)


def chat_page() -> None:
    """Render the chat page."""
    st.title("Chat")
    st.sidebar.markdown(f"{chatbot.get_model_name()}")
    display_chat_history()
    if user_prompt := st.chat_input("Type your message here..."):
        user_message = HumanMessage(user_prompt)
        display_message(user_message)
        with st.spinner("Thinking..."):
            try:
                response = chatbot.get_response(user_message)
                if response:
                    display_message(response)
            except Exception as e:
                st.error(f"Error getting response: {e}")    


if chatbot is not None:
    chat_page()
else:
    st.error("Chatbot is not initialized.")

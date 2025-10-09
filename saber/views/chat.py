"""This module implements the main chat page for the application, providing the
user interface for interacting with the chatbot. It handles message display and
user input processing.

Components:
    - Message display system for HumanMessage and AIMessage types
    - Chat history rendering with conversation persistence
    - User input processing with real-time responses
    - Error handling with user-friendly messages

Note:
    This module assumes the chatbot is properly initialized in the session
    state by the main application. If the chatbot is not available, an error
    message is displayed to guide the user.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage


# Retrieve chatbot instance from Streamlit session state
# This is initialized by the main application (app.py) when the session starts
chatbot = st.session_state.get("chatbot", None)


def display_message(message: HumanMessage | AIMessage) -> None:
    """Display a chat message with appropriate role-based styling.
    
    Renders a message in the Streamlit chat interface using role-specific
    styling (user vs assistant). The message content is displayed using
    Markdown formatting for rich text support.
    
    Args:
        message (HumanMessage | AIMessage): The message to display. Must be
            either a LangChain HumanMessage (from user) or AIMessage (from AI).

    Example:
        >>> from langchain_core.messages import HumanMessage, AIMessage
        >>> 
        >>> user_msg = HumanMessage("Hello, how are you?")
        >>> display_message(user_msg)  # Shows with user styling
        >>> 
        >>> ai_msg = AIMessage("I'm doing well, thank you!")
        >>> display_message(ai_msg)    # Shows with assistant styling
    
    Error Handling:
        If an invalid message type is provided, displays an error message
        in the UI instead of crashing the application.
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
    """Display the complete chat conversation history.
    
    Retrieves and renders all previous messages from the current chatbot
    session, maintaining the chronological order of the conversation.
    Each message is displayed with appropriate role-based styling.
        
    Performance Notes:
        - History length grows with conversation duration
        - Large histories may impact rendering performance
    """
    chat_history = chatbot.get_chat_history()
    for message in chat_history:
        display_message(message)


def chat_page() -> None:
    """Render the main chat interface page.
    
    Creates and manages the complete chat user interface, including the
    conversation display, user input handling, and response processing.
    This is the primary function that orchestrates the entire chat experience.
    
    Functionality:
        - Displays existing chat history on page load
        - Handles new user input via chat input widget
        - Processes messages through the chatbot
        - Shows real-time responses with loading indicators
        - Manages error states
    
    User Interaction Flow:
        1. User sees existing conversation history
        2. User types message in input field
        3. Message is immediately displayed in chat
        4. Loading spinner appears while processing
        5. AI response is generated and displayed
        6. Input field clears for next message
    """
    st.title("Chat")    
    # Display current model information
    st.sidebar.markdown(f"{chatbot.get_model_name()}")
    display_chat_history()    
    # User input handling: Process new messages
    if user_prompt := st.chat_input("Type your message here..."):
        # Create and display user message immediately
        user_message = HumanMessage(user_prompt)
        display_message(user_message)        
        # Generate AI response with loading indicator
        with st.spinner("Thinking..."):
            try:
                response = chatbot.get_response(user_message)
                if response:
                    display_message(response)
            except Exception as e:
                # Show user-friendly error message for any failures
                st.error(f"Error getting response: {e}")


# Main execution: Initialize chat page or show error
if chatbot is not None:
    chat_page()
else:
    st.error("Chatbot is not initialized.")

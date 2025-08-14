import os
import getpass as gp
import streamlit as st

from langchain.chat_models import init_chat_model
from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
)
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

class Chatbot:
    """Implements the chatbot user interface and functionalities

    Attributes:
        name: A string of the chatbot's name.
    """

    def __init__(self, name: str = "S.A.B.E.R."):
        """Initializes the chatbot with a name and does the initial setup.
        
        Args:
            name: Defines the name of the chatbot. (Default: "S.A.B.E.R.")
        """
        self.name = name
        # For security reasons, we do not store the API key in the code or in
        # any file like a ".env" or a streamlit "secrets.toml". The user must
        # enter the API key manually when prompted or set it as an environment 
        # variable.
        #
        # Proper way to set the API key as an environment variable:
        #   $ read -s GOOGLE_API_KEY
        #   $ export GOOGLE_API_KEY
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = gp.getpass(
                "Please, enter your Google API key: "
            )
        self.init_model()
        if not st.session_state.get("messages"):
            st.session_state.messages = []

    def init_model(self):
        """Sets up the model for the chatbot."""
        # Initialize the model if not already set
        if not st.session_state.get("model"):
            st.session_state.model = init_chat_model(
                model_name="gemini-2.5-flash",
                model_provider="google_genai",
            )
        # Initialize the workflow if not already set
        if not st.session_state.get("workflow"):
            workflow = StateGraph(state_schema=MessagesState)
            workflow.add_edge(START, "model")
            workflow.add_node("model", self.call_model)
            st.session_state.workflow = workflow
        # Compile the workflow if not already compiled
        if not st.session_state.get("model_app"):
            # Save the model state in memory
            memory = MemorySaver()
            model_app = workflow.compile(checkpointer=memory)
            st.session_state.model_app = model_app
        # Initialize the model_config if not already set
        if not st.session_state.get("model_config"):
            st.session_state.model_config = {
                "configurable": {
                    "thread_id": "abc123",
                }
            }

    def call_model(self, state: MessagesState) -> dict:
        """Calls the model with the current messages and returns the response.
        
        Args:
            state: The current state of the messages.

        Returns:
            A dictionary containing the response from the model.
        """
        response = st.session_state.model.invoke(state["messages"])
        return {"messages": response}        

    def show_message_history(self):
        """Displays the message history."""
        for message in st.session_state.messages:
            self.show_message(message)

    def get_user_prompt(self) -> HumanMessage | None:
        """Gets user's prompt from the chat interface.
        
        Returns:
            A HumanMessage containing the user's prompt.
        """
        if prompt := st.chat_input():
            message = HumanMessage(prompt)
            st.session_state.messages.append(message)
            return message
        return None
    
    def show_message(self, message: HumanMessage | AIMessage):
        """Displays a message in the chat interface.
        
        Args:
            message: The message to display.
        """
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    def get_assistant_response(self) -> AIMessage | None:
        """Gets assistant's response based on the message history.

        Returns:
            A AIMessage of the assistant's response.
        """
        messages = st.session_state.messages
        response = st.session_state.model_app.invoke(
            {"messages": messages},
            st.session_state.model_config
        )
        if response and "messages" in response:
            message = response["messages"][-1]
            messages.append(message)
            return message
        return None

    def run(self):
        """Run the chatbot."""
        st.title(f"{self.name}")
        self.show_message_history()
        prompt = self.get_user_prompt()
        if prompt:
            self.show_message(prompt)
            response = self.get_assistant_response()
            if response:
                self.show_message(response)

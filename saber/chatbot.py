import streamlit as st

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
        if not st.session_state.get("messages"):
            st.session_state.messages = []

    def show_message_history(self):
        """Displays the message history."""
        if len(st.session_state.messages) == 0:
            with st.chat_message("assistant"):
                st.markdown("Hello! How can I assist you today?")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def get_user_prompt(self) -> str:
        """Gets user's prompt from the chat interface.
        
        Returns:
            A string of the user's prompt.
        """
        if prompt := st.chat_input():
            return prompt
        return ""

    def run(self):
        """Run the chatbot."""
        st.title(f"{self.name}")
        self.show_message_history()
        prompt = self.get_user_prompt()

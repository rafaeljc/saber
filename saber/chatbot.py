import os
import getpass as gp
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
        # For security reasons, we do not store the API key in the code or in
        # any file like a ".env" or a streamlit "secrets.toml". The user must
        # enter the API key manually when prompted or set it as an environment 
        # variable.
        #
        # Proper way to set the API key as an environment variable:
        #   $ read -s GOOGLE_GENAI_API_KEY
        #   $ export GOOGLE_GENAI_API_KEY
        if not os.environ.get("GOOGLE_GENAI_API_KEY"):
            os.environ["GOOGLE_GENAI_API_KEY"] = gp.getpass(
                "Please, enter your Google GenAI API key: "
            )
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
    
    def show_message(self, role: str, content: str):
        """Displays a message in the chat interface.
        
        Args:
            role: The role of the message sender ("user" or "assistant").
            content: The content of the message.

        Raises:
            ValueError: If the role is not "user" or "assistant".
        """
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be either 'user' or 'assistant'.")
        st.session_state.messages.append({"role": role, "content": content})
        with st.chat_message(role):
            st.markdown(content)

    def get_assistant_response(self, prompt: str) -> str:
        """Gets assistant's response based on the user's prompt.
        
        Args:
            prompt: The user's prompt.

        Returns:
            A string of the assistant's response.
        """
        return f"Echo: {prompt}" 

    def run(self):
        """Run the chatbot."""
        st.title(f"{self.name}")
        self.show_message_history()
        prompt = self.get_user_prompt()
        if prompt:
            self.show_message(role="user", content=prompt)
            response = self.get_assistant_response(prompt)
            if response:
                self.show_message(role="assistant", content=response)

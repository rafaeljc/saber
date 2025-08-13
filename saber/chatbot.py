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

    def run(self):
        """Run the chatbot."""
        st.title(f"{self.name}")

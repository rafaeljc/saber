"""This module serves as the entry point for the application.

Application features:
- Web-based chat interface using Streamlit
- Command-line launcher for easy startup
- Comprehensive logging for debugging and monitoring

Usage:
    Command line (after pip install):
        $ saber
    
    Direct execution:
        $ python -m saber.app
        $ streamlit run saber/app.py

Requirements:
    - streamlit: Web framework for the chat interface
    - langchain: Language model integration
"""

import logging
import subprocess
import streamlit as st
from saber import Chatbot
from saber.routes import routes


# Configure application-wide logging
# Logs are written to both file (app.log) and console for comprehensive
# monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Persistent file logging
        logging.StreamHandler(),         # Console output
    ]
)


def run() -> None:
    """Run the application from the command line interface.

    Launches the Streamlit web application using subprocess, enabling users
    to start the chatbot interface by running the 'saber' command after
    installation. This function serves as the console script entry point.

    Usage:
        After pip installation:
            $ pip install .
            $ saber
        
        Direct module execution:
            $ python -m saber.app
    """
    try:
        subprocess.run(["streamlit", "run", __file__])
    except KeyboardInterrupt:
        print("Application stopped by user.")


def main() -> None:
    """Initialize and run the main Streamlit application.
    
    Sets up the application environment by:
    - Initializing the chatbot instance in Streamlit session state
    - Configuring navigation routing system
    - Starting the web application interface
    
    Session State Management:
        - Creates a single Chatbot instance per user session
        - Persists chatbot state across page interactions
        - Prevents re-initialization on subsequent function calls
    
    Navigation:
        Uses the routes configuration to enable multi-page navigation
        within the Streamlit application interface.
    """
    # Initialize chatbot instance in session state if not already present
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = Chatbot()    
    # Run the navigation system with configured routes
    st.navigation(routes).run()


if __name__ == "__main__":
    main()

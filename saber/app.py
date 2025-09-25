import logging
import subprocess
import streamlit as st
from saber import Chatbot
from saber.routes import routes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ]
)


def run():
    """Run the application from the command line.

    Enables the user to start the application by running the 'saber' command in
    the terminal after installing it with pip.

    Usage example:
        $ pip install .
        $ saber
    """
    try:
        subprocess.run(["streamlit", "run", __file__])
    except KeyboardInterrupt:
        print("Application stopped by user.")


def main():
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = Chatbot()
    st.navigation(routes).run()


if __name__ == "__main__":
    main()

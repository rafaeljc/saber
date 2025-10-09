"""This module defines the navigation structure for the application, configuring
the available pages and their properties for the multi-page interface.

Route Structure:
    Each route is defined as a Streamlit Page object with:
    - File path: Location of the page implementation
    - Title: Display name in the navigation menu
    - Icon: Visual identifier for the page (emoji or icon)

Usage:
    The routes tuple is imported and used by the main application:
    
    ```python
    from saber.routes import routes
    st.navigation(routes).run()
    ```

Note:
    Page files must be located relative to the main application directory.
    Icons can be emojis or any supported Streamlit icon format.
"""

import streamlit as st
from typing import Tuple


# Application Routes Configuration
routes = (
    # Main chat interface - primary user interaction page
    st.Page(
        "views/chat.py",
        title="Chat",
        icon="💬",
    ),    
    # Configuration page - chatbot settings and parameters
    st.Page(
        "views/settings.py",
        title="Settings",
        icon="⚙️",
    ),
)

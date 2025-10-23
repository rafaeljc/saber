"""This module provides a interface for uploading, viewing, and managing files
that can be processed by the chatbot. It handles file operations through the
chatbot's file storage system with proper error handling.

Supported File Types:
    - Text files (.txt)
    - PDF documents (.pdf)
    - Word documents (.docx)

Example Flow:
    1. User navigates to Files page
    2. Selects a file using the upload widget
    3. File is validated and stored via chatbot instance
    4. Success message displayed, file appears in management interface
    5. User can view uploaded files and optionally delete them

Note:
    This module requires the chatbot to be properly initialized in the session
    state. All configuration changes are applied immediately and persist
    throughout the session.
"""

import streamlit as st


# Retrieve chatbot instance from Streamlit session state
# This is initialized by the main application (app.py) when the session starts
chatbot = st.session_state.get("chatbot", None)


def upload_new_file() -> None:
    """Display file upload interface and handle file saving.

    Layout:
        **File Upload**
        - Single file upload widget
        - Supports: .txt, .pdf, .docx

    Note:
        This function handles only single file uploads to avoid Streamlit's
        problematic behavior with multiple file uploads, where the widget
        may retrigger upload operations during app reruns when users dismiss
        dialog boxes after successful uploads.
    """
    st.subheader("📤 Upload New File")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["txt", "pdf", "docx"]
    )
    if uploaded_file:
        filename = uploaded_file.name
        try:
            file = (filename, bytes(uploaded_file.getbuffer()))
            chatbot.write_uploaded_files([file])
        except Exception as e:
            st.error(f"Error uploading file: {e}")
        else:
            st.success(f"{filename} uploaded successfully!")


def show_uploaded_files() -> None:
    """Display uploaded files management interface with deletion capabilities.

    Layout:
        **Uploaded Files**
        - List of uploaded files with checkboxes for selection
        - Delete button to remove selected files

    Note:
        Uses forms widget to prevent unwanted app reruns when users interact
        with checkboxes. The form submit button ensures operations only occur
        when explicitly triggered by the user.
    """
    st.subheader("📎 Uploaded Files")
    uploaded_files = chatbot.get_uploaded_files_list()
    if uploaded_files:
        with st.form("uploaded_files_form"):
            files_to_delete = []
            for filename in uploaded_files:
                checkbox_key = f"delete_checkbox_{filename}"
                if st.checkbox(filename, key=checkbox_key):
                    files_to_delete.append(filename)
            successful_deletion_msg = st.session_state.get(
                "successful_deletion_msg", None
            )
            if successful_deletion_msg:
                st.success(successful_deletion_msg)
                st.session_state["successful_deletion_msg"] = None
            form_button = st.form_submit_button("Delete Selected Files")
            if form_button and files_to_delete:
                try:
                    chatbot.delete_uploaded_files(files_to_delete)
                    st.session_state["successful_deletion_msg"] = (
                        f"{len(files_to_delete)} files deleted successfully!"
                    )
                    # Rerun to update the uploaded files after deletion
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting files: {e}")
            elif form_button and not files_to_delete:
                st.warning("No files selected for deletion.")
    else:
        st.info("No files uploaded yet.")


def files_page() -> None:
    """Display the complete file management interface."""
    st.title("Files")
    upload_new_file()
    show_uploaded_files()


# Main execution: Initialize settings page or show error
if chatbot is not None:
    files_page()
else:
    st.error("Chatbot is not initialized.")

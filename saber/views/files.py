import streamlit as st


# Retrieve chatbot instance from Streamlit session state
# This is initialized by the main application (app.py) when the session starts
chatbot = st.session_state.get("chatbot", None)


def upload_new_file() -> None:
    """Display file upload interface and handle file saving.
    
    Note:
        Because Streamlit's file_uploader widget has a bad behavior on handling
        multiple file uploads in certain contexts (i.e. Streamlit reruns the app
        and tries to upload the same files again when the user closes any dialog
        box opened after a successful upload), this function handles only single
        file uploads at a time.
    """
    st.subheader("📤 Upload New File")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["txt", "pdf", "docx"])
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
    """Display the uploaded files and delete them."""
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
                "successful_deletion_msg", None)
            if successful_deletion_msg:
                st.success(successful_deletion_msg)
                st.session_state["successful_deletion_msg"] = None
            form_button = st.form_submit_button("Delete Selected Files")
            if form_button and files_to_delete:
                try:
                    chatbot.delete_uploaded_files(files_to_delete)
                    st.session_state["successful_deletion_msg"] = (
                        f"{len(files_to_delete)} files deleted successfully!")
                    # Rerun to update the uploaded files after deletion
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting files: {e}")
            elif form_button and not files_to_delete:
                st.warning("No files selected for deletion.")
    else:
        st.info("No files uploaded yet.")


def files_page() -> None:
    """Display the uploaded files management interface."""
    st.title("Files")
    upload_new_file()
    show_uploaded_files()


# Main execution: Initialize settings page or show error
if chatbot is not None:
    files_page()
else:
    st.error("Chatbot is not initialized.")

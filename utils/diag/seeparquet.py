import streamlit as st
import pandas as pd
import os

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(layout="wide", page_title="Parquet Data Viewer")

    st.title("Parquet File Data Viewer")
    st.write("Upload a Parquet file to view its contents in a table.")

    # File uploader allows user to upload their own parquet file
    uploaded_file = st.file_uploader("Choose a .parquet file", type="parquet")

    if uploaded_file is not None:
        try:
            # Read the uploaded parquet file into a pandas DataFrame
            df = pd.read_parquet(uploaded_file)

            st.success(f"Successfully loaded `{uploaded_file.name}`")
            st.write("### DataFrame Preview")

            # Display the DataFrame in the Streamlit app
            st.dataframe(df)

            # --- Optional: Display DataFrame Info ---
            st.write("### DataFrame Info")
            # Create a buffer to capture the output of df.info()
            from io import StringIO
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            # -----------------------------------------

        except Exception as e:
            st.error(f"Error loading or reading the file: {e}")
    else:
        st.info("Awaiting file upload.")

if __name__ == "__main__":
    main()

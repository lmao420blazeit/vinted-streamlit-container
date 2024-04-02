import streamlit as st
import json 
import os

def css_to_markdown(css_file):
    """
    Read a .css file and write its contents as Streamlit markdown.

    Parameters:
    - css_file (str): Path to the .css file.

    Returns:
    - markdown (str): Streamlit markdown containing the CSS code.
    """
    try:
        with open(css_file, 'r') as f:
            css_content = f.read()
        markdown = f"<style>{css_content}</style>"
        return st.write(markdown, 
                        unsafe_allow_html=True)
    except FileNotFoundError:
        return "File not found."


def load_credentials(path = "aws_rds_credentials.json"):
     with open(path, 'r') as file:
          config = json.load(file)

     # set up credentials
     for key in config.keys():
          os.environ[key] = config[key]

     return
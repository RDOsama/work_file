# import streamlit as st


# st.text_input("Type something", key="user_input")

# def on_click():
#     st.session_state.user_input = ""

# st.button("Clear", on_click=on_click)


import streamlit as st

# Create a placeholder for the input field
input_placeholder = st.empty()

# Add the input field to the placeholder
input_placeholder.text_input("Type something", key="user_input")

def on_click():
    # Clear the input field by emptying the placeholder
    input_placeholder.empty()

st.button("Clear", on_click=on_click)

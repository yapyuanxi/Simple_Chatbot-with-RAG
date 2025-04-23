# frontend.py

import streamlit as st
from backend import deepseek_rag_pipeline

st.title("DeepSeek R1 Chatbot")

# Initialize conversation history if not already in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept new user input
if prompt := st.chat_input("Type your message here..."):
    # Append and display user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant's response and process it
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            full_response = deepseek_rag_pipeline(prompt)
            
            # Extract only the final response after </think>
            if "</think>" in full_response:
                response = full_response.split("</think>")[-1].strip()
            else:
                response = full_response.strip()
            
            # Add debug toggle in sidebar if needed
            if st.sidebar.checkbox("Show debug info", False):
                st.write("Full response (including thinking):", full_response)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
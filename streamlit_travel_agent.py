
import streamlit as st
from wg import build_travel_agent_graph  # Your full logic lives in wg.py

# Initialize graph and session state
if 'graph' not in st.session_state:
    st.session_state.graph = build_travel_agent_graph()

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'state' not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "travel_details": {},
        "flight_data": None,
        "hotel_data": None,
        "itinerary": None,
        "error": None,
        "next": None
    }

st.title("🌍 Travel Booking Assistant")




# --- Input Form ---
with st.form("user_input_form"):
    user_input = st.text_area("Describe your travel plan:", height=100, placeholder="e.g., I need a flight from Mumbai to Delhi on 2025-06-10 returning on 2025-06-15 for 2 passengers")
    submitted = st.form_submit_button("Submit")

if submitted and user_input:
    # Add user message
    st.session_state.state["messages"].append({"role": "human", "content": user_input})

    # Run the LangGraph agent
    result = st.session_state.graph.invoke(st.session_state.state)

    # Update session state
    st.session_state.state = result


# --- Clear Button ---
if st.button("Clear Chat"):
    st.session_state.state = {
        "messages": [],
        "travel_details": {},
        "flight_data": None,
        "hotel_data": None,
        "itinerary": None,
        "error": None,
        "next": None
    }
    st.rerun()  # Use this for newer Streamlit versions


# --- Display Chat History ---
for msg in st.session_state.state["messages"]:
    if msg["role"] == "human":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "ai":
        st.markdown(
            f"<div style='font-size: 11px; white-space: pre-wrap;'><strong>Travel Assistant:</strong><br>{msg['content']}</div>",
            unsafe_allow_html=True
        )

# --- Optional Debug Info ---
with st.expander("🔍 Debug Info"):
    st.json(st.session_state.state)

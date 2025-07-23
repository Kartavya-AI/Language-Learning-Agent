import streamlit as st

def render_sidebar():
    st.sidebar.header("⚙️ AI & API Settings")
    key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.get("GEMINI_API_KEY", ""))
    if st.sidebar.button("💾 Save API Key"):
        if key:
            st.session_state["GEMINI_API_KEY"] = key
            st.success("✅ API Key saved!")
        else:
            st.error("❌ API Key missing!")

    return st.session_state.get("GEMINI_API_KEY")

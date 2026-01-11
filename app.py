"""
Math Mentor - Local Development Version
========================================
Simple Streamlit app using Agno with Gemini + Groq

Run: streamlit run app.py
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv

# ============================================
# LOAD ENV
# ============================================
load_dotenv()

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Math Mentor",
    page_icon="🧮",
    layout="wide"
)

# ============================================
# IMPORTS
# ============================================
from config import Config
from models.schemas import RawInput, InputType
from processors.text_processor import TextProcessor
from processors.ocr_processor import OCRProcessor
from processors.audio_processor import AudioProcessor
from agents.team import MathMentorTeam
from memory.store import MemoryStore

# ============================================
# CACHED LOADERS
# ============================================

@st.cache_resource
def load_team():
    return MathMentorTeam()

@st.cache_resource
def load_memory():
    return MemoryStore(Config.MEMORY_DB_PATH)

@st.cache_resource
def load_text_processor():
    return TextProcessor()

@st.cache_resource
def load_ocr():
    return OCRProcessor()

@st.cache_resource
def load_audio():
    return AudioProcessor()

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("🧮 Math Mentor")
    st.markdown("### AI-Powered Math Problem Solver")
    st.markdown("---")

    # ============================================
    # API CHECK
    # ============================================
    if Config.LLM_PROVIDER == "gemini" and not Config.GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found")
        st.stop()

    if Config.LLM_PROVIDER == "groq" and not Config.GROQ_API_KEY:
        st.error("GROQ_API_KEY not found")
        st.stop()

    # ============================================
    # INIT COMPONENTS
    # ============================================
    team = load_team()
    text_processor = load_text_processor()

    # ============================================
    # SESSION STATE
    # ============================================
    if "result" not in st.session_state:
        st.session_state.result = None

    if "pending_resubmit" not in st.session_state:
        st.session_state.pending_resubmit = False

    if "resubmitted_text" not in st.session_state:
        st.session_state.resubmitted_text = None

    # ============================================
    # HANDLE HITL RESUBMISSION
    # ============================================
    if st.session_state.pending_resubmit:
        with st.spinner("Re-solving clarified problem..."):
            raw_input = text_processor.process(
                st.session_state.resubmitted_text
            )
            st.session_state.result = team.solve(raw_input)

        st.session_state.pending_resubmit = False
        st.session_state.resubmitted_text = None

    # ============================================
    # SIDEBAR
    # ============================================
    with st.sidebar:
        st.header("Math Mentor")
        st.markdown("---")
        st.subheader("Status")

        if Config.LLM_PROVIDER == "gemini":
            st.success("Gemini connected")
        else:
            st.success("Groq connected")

        st.markdown("---")
        st.subheader("Agents")

        team_info = team.get_team_info()
        for agent in team_info.get("agents", []):
            st.caption(f"• {agent['name']}")

    # ============================================
    # INPUT TABS
    # ============================================
    tab1, tab2, tab3 = st.tabs(["Text", "Image", "Audio"])

    # ---------------- TEXT ----------------
    with tab1:
        st.subheader("Type your math problem")

        problem = st.text_area(
            "Enter problem:",
            height=120,
            placeholder="e.g., 3 * {4 [85 + 5 − (15 / 3)] + 2}"
        )

        if st.button("Solve", type="primary"):
            if problem.strip():
                with st.spinner("Solving..."):
                    raw_input = text_processor.process(problem)
                    st.session_state.result = team.solve(raw_input)
            else:
                st.warning("Please enter a problem")

    # ---------------- IMAGE ----------------
    with tab2:
        uploaded = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded:
            st.image(uploaded, caption="Uploaded", width=400)

            if st.button("Extract & Solve", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                    f.write(uploaded.getbuffer())
                    path = f.name

                try:
                    with st.spinner("Extracting..."):
                        raw_input = load_ocr().process(path)

                    if raw_input and raw_input.content:
                        with st.spinner("Solving..."):
                            st.session_state.result = team.solve(raw_input)
                    else:
                        st.error("OCR failed")
                finally:
                    os.unlink(path)

    # ---------------- AUDIO ----------------
    with tab3:
        if not Config.GROQ_API_KEY:
            st.warning("GROQ_API_KEY required for audio")
        else:
            audio_file = st.file_uploader(
                "Upload audio",
                type=["mp3", "wav", "m4a"]
            )

            if audio_file:
                st.audio(audio_file)

                if st.button("Transcribe & Solve", type="primary"):
                    suffix = Path(audio_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                        f.write(audio_file.getbuffer())
                        path = f.name

                    try:
                        with st.spinner("Transcribing..."):
                            raw_input = load_audio().process(path)

                        if raw_input and raw_input.content:
                            with st.spinner("Solving..."):
                                st.session_state.result = team.solve(raw_input)
                        else:
                            st.error("Transcription failed")
                    finally:
                        os.unlink(path)

    # ============================================
    # DISPLAY RESULT
    # ============================================
    if st.session_state.result:
        result = st.session_state.result
        st.markdown("---")

        # ---------- ERROR ----------
        if result.status == "error":
            st.error(result.error_message or "An unexpected error occurred")

        # ---------- HITL ----------
        elif result.status == "needs_hitl":
            st.warning(result.hitl_reason or "Clarification required")

            edited = st.text_area(
                "Clarify your problem:",
                value=result.parsed_problem.problem_text,
                key="hitl_input"
            )

            if st.button("Resubmit", key="resubmit_btn"):
                st.session_state.pending_resubmit = True
                st.session_state.resubmitted_text = edited
                st.rerun()

        # ---------- SUCCESS ----------
        elif result.status == "success":
            st.success(f"Answer: {result.solution.final_answer}")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### Explanation")
                st.info(result.explanation.summary)

                if result.explanation.detailed_steps:
                    with st.expander("Detailed Steps", expanded=True):
                        for i, step in enumerate(
                            result.explanation.detailed_steps, 1
                        ):
                            st.markdown(f"Step {i}: {step}")

            with col2:
                st.markdown("### Verification")
                conf = result.verification.confidence

                if conf >= 0.8:
                    st.success(f"Confidence: {conf:.0%}")
                elif conf >= 0.6:
                    st.warning(f"Confidence: {conf:.0%}")
                else:
                    st.error(f"Confidence: {conf:.0%}")

            st.markdown("---")
            if st.button("New Problem"):
                st.session_state.result = None
                st.rerun()

    st.markdown("---")
    st.caption("Math Mentor | Agno + Gemini + Groq")

# ============================================
if __name__ == "__main__":
    main()

"""
Math Mentor - Local / Deployable Streamlit App
==============================================
AI-powered multimodal JEE-style math solver
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
from processors.text_processor import TextProcessor
from processors.ocr_processor import OCRProcessor
from processors.audio_processor import AudioProcessor
from agents.team import MathMentorTeam
from memory.store import MemoryStore
from rag.auto_ingest import auto_ingest_if_needed

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
# AUTO INGEST KB (RUN ONCE)
# ============================================

@st.cache_resource
def ensure_kb_loaded():
    auto_ingest_if_needed()

# ============================================
# MAIN APP
# ============================================

def main():

    ensure_kb_loaded()

    st.title("🧮 Math Mentor")
    st.markdown("### Reliable AI Math Solver (RAG + Agents + HITL + Memory)")
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
    memory = load_memory()
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

    if "user_correction" not in st.session_state:
        st.session_state.user_correction = ""

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

        if Config.LLM_PROVIDER == "gemini":
            st.success("Gemini connected")
        else:
            st.success("Groq connected")

        st.markdown("---")
        st.subheader("Agents")
        for agent in team.get_team_info().get("agents", []):
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
            placeholder="e.g., If x^2 - 5x + 6 = 0, find x"
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
            st.image(uploaded, caption="Uploaded image", width=400)

            if st.button("Extract & Solve", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                    f.write(uploaded.getbuffer())
                    path = f.name

                try:
                    with st.spinner("Extracting text..."):
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
            st.warning("GROQ_API_KEY required for audio input")
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
            st.warning(f"HUMAN REVIEW REQUIRED: {result.hitl_reason}")

            edited = st.text_area(
                "Clarify your problem:",
                value=result.parsed_problem.problem_text,
                key="hitl_input"
            )

            if st.button("Resubmit"):
                st.session_state.pending_resubmit = True
                st.session_state.resubmitted_text = edited
                st.rerun()

        # ---------- SUCCESS ----------
        elif result.status == "success":
            st.success(f"Answer: {result.solution.final_answer}")

            col1, col2 = st.columns([2, 1])

            # -------- Explanation --------
            with col1:
                st.markdown("### Explanation")
                st.info(result.explanation.summary)

                if result.explanation.detailed_steps:
                    with st.expander("Detailed Steps", expanded=True):
                        for i, step in enumerate(result.explanation.detailed_steps, 1):
                            st.markdown(f"**Step {i}:** {step}")

                # -------- RAG CONTEXT --------
                if result.solution.context_used:
                    with st.expander("📚 Retrieved Reference Material"):
                        for i, ctx in enumerate(result.solution.context_used, 1):
                            st.markdown(f"**Source {i}: {ctx.source}**")
                            st.caption(f"Relevance: {ctx.relevance_score:.2f}")
                            st.write(ctx.text)
                            st.markdown("---")

            # -------- Verification --------
            with col2:
                st.markdown("### Verification")
                conf = result.verification.confidence

                if conf >= 0.8:
                    st.success(f"Confidence: {conf:.0%}")
                elif conf >= 0.6:
                    st.warning(f"Confidence: {conf:.0%}")
                else:
                    st.error(f"Confidence: {conf:.0%}")

                with st.expander("🧠 Agent Trace"):
                    st.markdown("- Parser Agent → structured problem")
                    st.markdown("- Router Agent → problem classification")
                    st.markdown("- Solver Agent → RAG-based reasoning")
                    st.markdown("- Verifier Agent → correctness checks")

            # -------- FEEDBACK --------
            st.markdown("---")
            st.markdown("### Was this solution correct?")

            col_ok, col_bad = st.columns(2)

            with col_ok:
                if st.button("✅ Correct"):
                    memory.store_success(
                        raw_input=result.raw_input,
                        parsed_problem=result.parsed_problem,
                        solution=result.solution,
                        verification=result.verification
                    )
                    st.success("Feedback saved. Thank you!")

            with col_bad:
                st.session_state.user_correction = st.text_area(
                    "Provide correction:",
                    value=st.session_state.user_correction
                )
                if st.button("❌ Submit Correction"):
                    memory.store_failure(
                        raw_input=result.raw_input,
                        parsed_problem=result.parsed_problem,
                        solution=result.solution,
                        correction=st.session_state.user_correction
                    )
                    st.warning("Correction stored for future learning.")

            if st.button("New Problem"):
                st.session_state.result = None
                st.session_state.user_correction = ""
                st.rerun()

    st.markdown("---")
    st.caption("Math Mentor | RAG + Agents + HITL + Memory")

# ============================================
if __name__ == "__main__":
    main()

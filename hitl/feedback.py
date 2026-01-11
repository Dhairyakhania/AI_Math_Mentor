import streamlit as st
from models.schemas import UserFeedback, FinalResult
from memory.store import MemoryStore
from typing import Optional


class HITLModule:
    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store

    # =============================
    # EXTRACTION REVIEW
    # =============================
    def review_extraction(
        self,
        extracted_text: str,
        confidence: float,
        input_type: str,
        interaction_id: str
    ) -> Optional[dict]:

        st.warning(
            f"⚠️ Low {input_type.upper()} confidence ({confidence:.1%}). Please review:"
        )

        corrected = st.text_area(
            "Extracted text (edit if needed):",
            extracted_text,
            height=150,
            key=f"extraction_text_{interaction_id}"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Approve", key=f"approve_extraction_{interaction_id}"):
                if corrected != extracted_text:
                    self.memory.save_ocr_correction(extracted_text, corrected)
                return {"approved": True, "text": corrected}

        with col2:
            if st.button("❌ Reject & Re-upload", key=f"reject_extraction_{interaction_id}"):
                return {"approved": False, "text": None}

        return None

    # =============================
    # SOLUTION FEEDBACK
    # =============================
    def review_solution(
        self,
        result: FinalResult,
        interaction_id: str
    ) -> Optional[UserFeedback]:

        st.divider()
        st.subheader("📝 Was this solution helpful?")

        # Prevent double submission
        feedback_key = f"feedback_submitted_{interaction_id}"
        if st.session_state.get(feedback_key):
            st.info("Feedback already submitted. Thank you!")
            return None

        choice = st.radio(
            "Select one:",
            ["Correct", "Partially Correct", "Incorrect"],
            key=f"feedback_choice_{interaction_id}"
        )

        comment = None
        corrected_solution = None

        if choice == "Partially Correct":
            comment = st.text_input(
                "What was wrong?",
                key=f"partial_comment_{interaction_id}"
            )

        elif choice == "Incorrect":
            corrected_solution = st.text_area(
                "Provide the correct solution:",
                height=150,
                key=f"correction_{interaction_id}"
            )
            comment = st.text_input(
                "Additional comments:",
                key=f"incorrect_comment_{interaction_id}"
            )

        if st.button("Submit Feedback", key=f"submit_feedback_{interaction_id}"):
            feedback = UserFeedback(
                feedback_type=choice.lower().replace(" ", "_"),
                comment=comment,
                corrected_solution=corrected_solution
            )

            self.memory.save_feedback(interaction_id, feedback)
            st.session_state[feedback_key] = True
            st.success("Thank you for your feedback!")

            return feedback

        return None

    # =============================
    # CLARIFICATION REQUEST
    # =============================
    def request_clarification(self, reason: str, parsed_problem, interaction_id: str) -> Optional[dict]:

        st.warning(f"⚠️ Clarification needed: {reason}")

        st.json({
            "problem": parsed_problem.problem_text,
            "topic": parsed_problem.topic.value,
            "variables": parsed_problem.variables,
            "constraints": parsed_problem.constraints
        })

        clarified = st.text_area(
            "Please clarify or rephrase the problem:",
            parsed_problem.problem_text,
            height=150,
            key=f"clarification_text_{interaction_id}"
        )

        if st.button("Submit Clarification", key=f"submit_clarification_{interaction_id}"):
            return {"clarified_text": clarified}

        return None

    # =============================
    # HITL BANNER
    # =============================
    def display_hitl_banner(self, reason: str):
        st.error(
            f"🤚 **Human Review Required**\n\nReason: {reason}\n\nPlease review below."
        )

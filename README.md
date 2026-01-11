# 🧮 Math Mentor – JEE-Grade AI Math Solver

Math Mentor is an **AI-powered, multi-agent mathematics problem solver** designed to handle **JEE-level algebra, calculus (including definite integrals), probability, and word problems** with **structured reasoning, verification, and student-friendly explanations**.

Built using the **Agno agent framework**, it combines **LLM reasoning** with **deterministic mathematical checks** to produce reliable, exam-grade solutions.

---

## ✨ Key Features

- ✅ **Multi-Agent Architecture** (Parser → Strategy → Router → Solver → Verifier → Explainer)
- 📐 **Advanced Calculus Support**
  - Differentiation
  - Indefinite & **definite integrals**
  - Natural-language limits (`from 2 to 5`, `(2 to 5)`, `[2,5]`, `∫₂⁵`)
- 📊 **Probability & Word Problems**
- 🧠 **JEE-style step-by-step reasoning**
- 🔍 **Deterministic Verification**
  - Substitution checks
  - Domain validation
  - Probability bounds
- 📖 **Equation-first explanations**
- 🧩 **HITL (Human-in-the-Loop) clarification flow**
- 🎙️ **Text, Image (OCR), and Audio input**
- ⚡ **Gemini or Groq LLM support**
- 🖥️ **Streamlit-based UI**

---

## 🏗️ System Architecture

User Input
│
▼
ParserAgent
│
▼
StrategyAgent
│
▼
RouterAgent
│
▼
SolverAgent
│
▼
VerifierAgent
│
▼
ExplainerAgent
│
▼
Final Answer + Explanation


Each agent has a **single responsibility**, making the system modular, debuggable, and extensible.

---

## 🤖 Agents Overview

### ParserAgent
- Validates raw input (text / image / audio)
- Detects algebra, calculus, probability, and word problems
- Normalizes calculus phrasing (limits, integrals)
- Outputs structured `ParsedProblem`

### StrategyAgent
- Determines problem type
- Plans the solution approach

### RouterAgent
- Selects solving strategy
- Extracts equations, variables, and constraints

### SolverAgent
- Produces **JEE-style structured steps**
- Handles algebra, calculus, and probability
- Evaluates **definite integrals numerically**
- Deterministic fallback avoids `"Unable to determine"`

### VerifierAgent
- Validates correctness
- Deterministic substitution for simple equations
- LLM-based verification for complex cases
- Assigns confidence score

### ExplainerAgent
- Cleans solver metadata
- Preserves equations
- Explains *why* each step is valid
- Produces student-friendly explanations

---

## 📥 Supported Input Types

| Input Type | Description |
|----------|-------------|
| 📝 Text | Direct math input |
| 📷 Image | OCR-based extraction |
| 🎤 Audio | Speech-to-text (Groq) |

---

## 📐 Example Problems Supported

- **Algebra**  
  `Solve x² - 5x + 6 = 0`

- **Differentiation**  
  `Find the derivative of 3x² + 4x - 2`

- **Definite Integrals**  
  `Evaluate the integral from 2 to 5 of (x³ - 2x² + x + 3) dx`

- **Probability**  
  `Probability of exactly two heads in three coin tosses`

- **Word Problems**  
  `A bag contains 5 red and 3 blue balls...`

---

## 🖥️ UI (Streamlit)

- Input tabs: Text / Image / Audio
- Live solution tracing
- HITL clarification prompts
- Confidence visualization

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd math-mentor
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here

# OR

LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_key_here

```

### 5️⃣ Run the app
```bash
streamlit run app.py
```

## 🧪 Reliability & Safety

- Deterministic math checks where possible

- Tool usage restricted to SolverAgent (Groq-safe)

- Graceful fallbacks — pipeline never crashes

- Confidence-based HITL escalation

## 👨‍💻 Use Cases

- JEE / competitive exam preparation

- Conceptual math learning

- Interview-ready AI system showcase

- Multi-agent LLM architecture reference

## 📜 License

For educational and research purposes.
You may adapt or extend with attribution.

## ⭐ Final Note

Math Mentor is not just a chatbot —
it is a structured reasoning system that mirrors human mathematical thinking, step by step.
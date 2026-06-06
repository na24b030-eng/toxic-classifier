import streamlit as st
import pickle
import re
import string
import numpy as np
import json
from nltk.stem import PorterStemmer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Toxic Content Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; }
    h1, h2, h3 { color: #e8e8f0; }
    .pred-box {
        padding: 24px;
        border-radius: 8px;
        text-align: center;
        margin: 16px 0;
    }
    .hate    { background: rgba(248,113,113,0.15); border: 1px solid #f87171; }
    .offense { background: rgba(251,146,60,0.15);  border: 1px solid #fb923c; }
    .neither { background: rgba(110,231,183,0.15); border: 1px solid #6ee7b7; }
    .metric-card {
        background: #11111a;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stTextArea textarea {
        background-color: #11111a !important;
        color: #e8e8f0 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('artifacts/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('artifacts/lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('artifacts/metrics.json', 'r') as f:
        metrics = json.load(f)
    return model, vectorizer, lr_model, metrics

model, vectorizer, lr_model, metrics = load_artifacts()

# ── Preprocessing ─────────────────────────────────────────────────────────────
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','then','once','here','there',
    'when','where','why','how','all','both','each','few','more','most','other',
    'some','such','than','too','very','s','t','just','should','now','rt'
}

stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def get_top_words(text, lr_model, vectorizer, pred_class, n=5):
    feature_names = vectorizer.get_feature_names_out()
    coef          = lr_model.coef_[pred_class]
    tokens        = text.split()
    word_scores   = []
    for token in tokens:
        indices = np.where(feature_names == token)[0]
        if len(indices) > 0:
            word_scores.append((token, coef[indices[0]]))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:n]

# ── Class config ──────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Hate Speech', 'Offensive Language', 'Neither']
CLASS_COLORS = ['#f87171', '#fb923c', '#6ee7b7']
CLASS_CSS    = ['hate', 'offense', 'neither']
CLASS_EMOJI  = ['🚨', '⚠️', '✅']
CLASS_DESC   = [
    'This content targets individuals or groups based on identity characteristics.',
    'This content contains offensive or abusive language but is not identity-targeted hate speech.',
    'This content does not appear to contain hate speech or offensive language.'
]

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ Toxic Content Classifier")
st.markdown("*Automated content moderation using NLP and Machine Learning*")
st.divider()

tab1, tab2 = st.tabs(["🔍 Classifier", "📊 Model Analytics"])

# ════════════════════════════════════════════════════════
# TAB 1 — CLASSIFIER
# ════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("### Enter Text")
        user_input = st.text_area(
            label="",
            placeholder="Paste any social media comment or text here...",
            height=160,
            label_visibility="collapsed"
        )

        # Example buttons
        st.markdown("**Try an example:**")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            if st.button("🚨 Hate example"):
                st.session_state['example'] = "These people should go back to where they came from, we don't want their kind here"
        with ex_col2:
            if st.button("⚠️ Offensive example"):
                st.session_state['example'] = "shut up you stupid idiot nobody asked for your opinion"
        with ex_col3:
            if st.button("✅ Neutral example"):
                st.session_state['example'] = "Had a wonderful time at the conference today, learned so much"

        if 'example' in st.session_state and not user_input:
            user_input = st.session_state['example']

        predict_btn = st.button("🔍 Analyse Text", type="primary", use_container_width=True)

    with col2:
        if predict_btn and user_input.strip():
            cleaned = clean_text(user_input)

            if cleaned.strip() == '':
                st.warning("Text is empty after cleaning. Please enter more content.")
            else:
                features  = vectorizer.transform([cleaned])
                pred_cls  = model.predict(features)[0]
                proba     = model.predict_proba(features)[0]

                # Prediction box
                css_class = CLASS_CSS[pred_cls]
                emoji     = CLASS_EMOJI[pred_cls]
                label     = CLASS_NAMES[pred_cls]
                color     = CLASS_COLORS[pred_cls]

                st.markdown(f"""
                <div class="pred-box {css_class}">
                    <div style="font-size:2.5rem">{emoji}</div>
                    <div style="font-size:1.4rem; font-weight:700;
                                color:{color}; margin:8px 0">{label}</div>
                    <div style="font-size:0.85rem; color:#6b6b85">
                        {CLASS_DESC[pred_cls]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence scores
                st.markdown("**Confidence Scores**")
                for i, (name, prob, clr) in enumerate(
                        zip(CLASS_NAMES, proba, CLASS_COLORS)):
                    st.markdown(f"<span style='color:{clr}'>{name}</span>",
                                unsafe_allow_html=True)
                    st.progress(float(prob), text=f"{prob*100:.1f}%")

                # Top influential words
                st.markdown("**Top Influential Words**")
                top_words = get_top_words(cleaned, lr_model, vectorizer, pred_cls)
                if top_words:
                    pills = ' '.join([
                        f"<span style='background:{CLASS_COLORS[pred_cls]}22;"
                        f"border:1px solid {CLASS_COLORS[pred_cls]};"
                        f"padding:3px 10px; border-radius:20px; font-size:0.85rem;"
                        f"color:{CLASS_COLORS[pred_cls]}; margin:3px'>{w}</span>"
                        for w, _ in top_words
                    ])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.caption("No strongly weighted words found for this input.")

                # Cleaned text expander
                with st.expander("See preprocessed text"):
                    st.code(cleaned)

        elif predict_btn:
            st.warning("Please enter some text first.")
        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#6b6b85'>
                <div style='font-size:3rem'>🛡️</div>
                <div style='margin-top:12px'>Enter text on the left and click Analyse</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance")

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class='metric-card'>
            <div style='color:#6b6b85;font-size:0.8rem'>BEST MODEL</div>
            <div style='color:#6ee7b7;font-size:1.1rem;font-weight:700;margin-top:6px'>Linear SVM</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class='metric-card'>
            <div style='color:#6b6b85;font-size:0.8rem'>ACCURACY</div>
            <div style='color:#6ee7b7;font-size:1.6rem;font-weight:700;margin-top:6px'>89.9%</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class='metric-card'>
            <div style='color:#6b6b85;font-size:0.8rem'>WEIGHTED F1</div>
            <div style='color:#6ee7b7;font-size:1.6rem;font-weight:700;margin-top:6px'>88.5%</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class='metric-card'>
            <div style='color:#6b6b85;font-size:0.8rem'>DATASET SIZE</div>
            <div style='color:#6ee7b7;font-size:1.6rem;font-weight:700;margin-top:6px'>24,780</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Model comparison table
    st.markdown("### Model Comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
| Model | Accuracy | Weighted F1 | Hate Recall |
|---|---|---|---|
| Naive Bayes | 86.3% | 83.8% | 8% ❌ |
| Logistic Regression | 85.4% | 86.9% | 62% ⚠️ |
| **Linear SVM** | **89.9%** | **88.5%** | **16%** ✅ |
        """)
    with col_b:
        st.markdown("""
**Why not accuracy alone?**

The dataset has 77% Offensive Language.
A model predicting *everything* as Offensive
would score 77% accuracy — but be useless.

Weighted F1 and per-class Recall are the
real indicators of a good moderation model.
        """)

    st.divider()

    # Visualisations
    st.markdown("### EDA & Results")
    img_tabs = st.tabs([
        "Class Distribution", "Text Length",
        "Vocabulary", "Word Clouds",
        "Confusion Matrix", "Feature Importance"
    ])

    import os
    def show_image(path, caption):
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
        else:
            st.warning(f"Image not found: {path}")

    with img_tabs[0]:
        show_image("images/class_distribution.png",
                   "Heavy class imbalance — Offensive dominates at 77%")
    with img_tabs[1]:
        show_image("images/text_length_eda.png",
                   "Text length distributions and boxplots by class")
    with img_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            show_image("images/top_words_before.png", "Before cleaning")
        with c2:
            show_image("images/top_words_after.png", "After cleaning")
    with img_tabs[3]:
        show_image("images/word_clouds.png",
                   "Word clouds per class after preprocessing")
    with img_tabs[4]:
        show_image("images/confusion_matrices.png",
                   "Confusion matrices for all three models")
    with img_tabs[5]:
        show_image("images/feature_importance.png",
                   "Top LR coefficients driving each class prediction")

    st.divider()
    st.markdown("### Limitations & Future Work")
    st.markdown("""
- **Hate Speech recall is low (16%)** — the minority class is hardest to detect.
  This is a known challenge with heavily imbalanced datasets.
- **Stemming loses morphology** — PorterStemmer is fast but crude;
  lemmatisation or subword tokenisation would improve edge cases.
- **Future:** Fine-tuned BERT or RoBERTa would capture context
  (e.g. *"I don't hate anyone"* vs *"I hate everyone"*) that
  bag-of-words models fundamentally cannot.
""")
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import os
from groq import Groq

# --- KONFIGURASI GROQ ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# --- DEFINISI CUSTOM LAYER ---
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        attention_weights = tf.nn.softmax(inputs, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        return super().get_config()

# ---LOAD DATA & MODEL ---
@st.cache_resource
def load_all_assets():
    try:
        model = tf.keras.models.load_model(
            "skill_recommendation_model.keras", 
            custom_objects={"AttentionLayer": AttentionLayer}
        )
        with open("mlb.pkl", "rb") as f:
            mlb = pickle.load(f)
        return model, mlb
    except Exception as e:
        st.error(f"Gagal memuat file model: {e}")
        return None, None

model, mlb = load_all_assets()

# --- PREDICT ---

def predict_skills(text):
    text_tensor = tf.constant([text])
    pred = model.predict(text_tensor)[0]
    
    # Ambil top 5 skill
    top_indices = np.argsort(pred)[-5:][::-1]
    skills = [mlb.classes_[i] for i in top_indices]
    scores = [float(pred[i]) for i in top_indices]
    
    return list(zip(skills, scores))

# --- GROQ RECOMMENDATION ---
@st.cache_data(show_spinner=False)
def get_groq_explanation(user_goal, skills_string):
    prompt = f'''
    Saya ingin menjadi {user_goal}.

    Skill saya saat ini (berdasarkan analisis deskripsi):
    {skills_string}

    Berikan:
    1. Roadmap belajar
    2. Skill tambahan yang perlu dipelajari
    3. Rekomendasi sertifikasi
    4. Saran karier
    
    Gunakan Bahasa Indonesia yang profesional dan format Markdown yang rapi.
    '''
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Gagal mendapatkan roadmap: {str(e)}"

# --- TAMPILAN DASHBOARD (UI) ---
st.set_page_config(page_title="Skill Recommender", layout="wide")
st.title("Skill Recommendation System")

# INPU USER
user_input = st.text_area(
    "Masukkan Nama Pekerjaan Impian Anda:", 
    placeholder="ex : Software Engineer",
    height=150
)

if st.button("Buat Rekomendasi"):
    if user_input.strip() == "":
        st.warning("Input tidak boleh kosong")
    elif model is None:
        st.error("Model gagal dimuat.")
    else:
        # PEOSES PREDICT
        results = predict_skills(user_input)
        skills_string = ", ".join([s for s, score in results])
        
        # LAYOUT JAWABAN SISTEM
        col1, col2 = st.columns([1, 1.3])
        
        with col1:
            st.subheader("Rekomendasi Skill")
            st.write("Skill yang direkomendasikan: ")
            for skill, score in results:
                st.write(f"**{skill}** ({score:.2f})")
                st.progress(score)
        
        with col2:
            st.subheader("Roadmap & Saran Karier")
            with st.spinner("Sedang menyusun roadmap belajar..."):
                # CMEMANGGIL GROQ
                explanation = get_groq_explanation(user_input, skills_string)
                st.markdown(explanation)
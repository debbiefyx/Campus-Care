import requests
import streamlit as st
import pymysql
import pickle
import joblib
import numpy as np
import pymysql.cursors
import time
import bcrypt
import os
import pandas as pd
import plotly.graph_objects as go
from pymysql.cursors import DictCursor
import streamlit.components.v1 as components
from dotenv import load_dotenv
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from datetime import datetime

load_dotenv()
print("✅ Loaded ENV:", os.getenv("MYSQL_USER"), "(pass hidden)")

ip = requests.get("https://api.ipify.org").text
st.markdown(f"**Public IP Address of this app:** `{ip}`")

def get_db_connection():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", 3306)),  # Default port if not specified
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASS"),
        database=os.getenv("MYSQL_DB"),
        cursorclass=pymysql.cursors.DictCursor
    )

def hash_password(password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)
    
def get_user_id(username):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            row = cursor.fetchone()
            return row["id"] if row else None

def validate_user(username, password):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
            row = cursor.fetchone()
            if row:
                # Verify the provided password against the stored hashed password
                stored_password = row["password"].encode('utf-8')  
                if verify_password(stored_password, password):
                    return True
    return False

def create_user(username, password):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return False  
            hashed_password = hash_password(password)
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
    return True

def assign_cluster(user_vector, group_label):
    df = pd.read_csv("all_cluster_profiles.csv")
    
    df_group = df[df["Group"] == group_label].copy()

    cluster_features = [
        "Coursework_Pressure", "Study_Hours_Per_Week", "Academic_Workload",
        "CoCurricular_Involvement", "Isolation_Frequency", "Physical_Activity_Freq",
        "Sleep_Hours_Per_Night", "Recent_Suicidal_Thoughts", "Financial_Stress", "Age"
    ]

    # Normalize user's data using MinMaxScaler 
    scaler = MinMaxScaler()
    scaler.fit(df_group[cluster_features])
    user_vector_scaled = scaler.transform([user_vector])

    # Compare to each cluster center
    distances = []
    for _, row in df_group.iterrows():
        cluster_center = row[cluster_features].values
        dist = euclidean(user_vector_scaled[0], cluster_center)
        distances.append(dist)

    # Get the index of the closest cluster
    min_idx = np.argmin(distances)
    return int(df_group.iloc[min_idx]["Cluster"])

def save_high_risk_response(user_id, age, study_hours, coursework_pressure, academic_workload,
                             sleep_hours, physical_activity, isolation, financial_stress,
                             cocurricular, suicidal_binary, prediction_result, cluster):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            query = """
                INSERT INTO high_risk_responses (
                    user_id, age, study_hours, coursework_pressure, academic_workload,
                    sleep_hours, physical_activity, isolation, financial_stress,
                    cocurricular, suicidal_thoughts, prediction_result, cluster
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                user_id, age, study_hours, coursework_pressure, academic_workload,
                sleep_hours, physical_activity, isolation, financial_stress,
                cocurricular, suicidal_binary, prediction_result, cluster
            ))
        conn.commit()

def save_self_check_visit(user_id, total_score, risk_level):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO self_check_logs (user_id, score, risk_level)
                VALUES (%s, %s, %s)
            """, (user_id, total_score, risk_level))
        conn.commit()

def get_self_check_stats(user_id):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM self_check_logs WHERE user_id = %s", (user_id,))
            total_row = cursor.fetchone()
            print("Total row returned:", total_row)
            total = total_row["count"] if total_row else 0

            cursor.execute("""
                SELECT COUNT(*) AS count FROM self_check_logs 
                WHERE user_id = %s AND risk_level = 'Low'
            """, (user_id,))
            low_row = cursor.fetchone()
            low = low_row["count"] if low_row else 0

            cursor.execute("""
                SELECT COUNT(*) AS count FROM self_check_logs 
                WHERE user_id = %s AND risk_level = 'High'
            """, (user_id,))
            high_row = cursor.fetchone()
            high = high_row["count"] if high_row else 0

    return total, low, high

def get_recent_clusters(user_id):
    cluster_df = pd.read_csv("all_cluster_profiles.csv")

    # Build a lookup: {(group_label, cluster_num): (friendly_name, description)}
    cluster_info = {}
    for _, row in cluster_df.iterrows():
        group_label = row["Group"]
        cluster_num = int(row["Cluster"])
        friendly_name = row.get("Cluster_Name", f"{group_label} Cluster {cluster_num}")
        description = row.get("Cluster_Description", "No description provided.")
        cluster_info[(group_label, cluster_num)] = (friendly_name, description)

    recent_clusters = []

    # Query the two most recent high-risk responses
    conn = get_db_connection()
    with conn:
        with conn.cursor(cursor=DictCursor) as cursor:
            cursor.execute("""
                SELECT cluster, prediction_result, submitted_at 
                FROM high_risk_responses 
                WHERE user_id = %s 
                ORDER BY submitted_at DESC 
                LIMIT 2
            """, (user_id,))
            results = cursor.fetchall()

    for row in results:
        cluster_num = int(row["cluster"])
        pred_result = row["prediction_result"]
        date_str = row["submitted_at"].strftime("%Y-%m-%d")

        # Map prediction_result to group name
        group = (
            "Moderate" if pred_result == 1 else
            "Severe" if pred_result == 2 else
            "Mild"
        )

        # Get custom name and description
        friendly_name, description = cluster_info.get((group, cluster_num), (f"{group} – Cluster {cluster_num}", "No description available."))

        recent_clusters.append({
            "label": friendly_name,
            "date": date_str,
            "description": description
        })

    return recent_clusters

def save_reflection(user_id, module_name, reflection):
    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO user_reflections (user_id, module_name, reflection)
                VALUES (%s, %s, %s)
            """, (user_id, module_name, reflection))
        conn.commit()


# ────────────────────────────────
# Streamlit Session & UI Setup
# ────────────────────────────────
st.set_page_config(page_title="Campus Care", layout="wide")

PAGES = {
    "🏠 Overview": "overview",
    "🧠 Self Assessment": "self_check",
    "🔴 High-Risk Pathway": "high_risk_pathway",
    "🟢 Low-Risk Pathway": "low_risk_pathway",
    "📘 Low-Risk Modules": "low_risk_modules",
    "📈 Dashboard": "dashboard"
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "overview"


#--- CSS Styling ---
st.markdown("""
<style>
/* General background and color theming */
[data-testid="stAppViewContainer"] > .main {
    background-image: url('https://i.pinimg.com/736x/31/fb/ef/31fbef452c9ce1f872c176c58a33c19e.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: #333;
}

/* Input fields and buttons */
.stTextInput input, .stNumberInput input {
    padding: 0.5rem;
    border-radius: 8px;
    width: 100% !important;
}

.stButton > button {
    background-color: #56999C !important;
    color: white !important;
    font-weight: bold;
    width: 100%;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin: 5px auto;
    display: block;
}

.feature-card {
    background-color: white;
    padding: 16px;
    border-radius: 10px;
    border-left: 5px solid #56999C;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 20px;
    position: relative;
    transition: all 0.3s ease-in-out;
}
.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
}
.step-circle {
    background-color: #56999C;
    color: white;
    font-weight: bold;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: absolute;
    top: -10px;
    left: -10px;
    font-size: 16px;
}

/* Chat animation for Kai */
[data-testid="stChatMessageContent"] {
    background-color: #f0fdfd !important;
    border-left: 4px solid #56999C !important;
    padding: 1rem !important;
    border-radius: 8px !important;
    margin: 10px 0 !important;
    animation: fadeInSlide 0.8s ease-in-out;
}

@keyframes fadeInSlide {
    0% {
        opacity: 0;
        transform: translateY(10px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

.metric-card {
    background-color: #C4E1E6;
    padding: 1.2rem;
    border: 1px solid #d3d3d3;
    border-radius: 12px;
    text-align: center;
    box-shadow: 1px 1px 5px rgba(0, 153, 255, 0.2);
    margin-bottom: 1rem;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
}

.cluster-card {
    background-color: #eef6f9;
    padding: 1rem;
    border-left: 5px solid #3399cc;
    border-radius: 10px;
    margin-bottom: 1rem;
</style>
""", unsafe_allow_html=True)

# --------------------------
#    Sidebar Navigation    
# --------------------------
with st.sidebar:
    if st.session_state.authenticated:
        if "username" in st.session_state:
            st.markdown(f"Hey **{st.session_state.username}**, welcome!")
            st.markdown("You're on your wellness journey 💙")

        st.markdown("---")

        st.markdown("## 📌 Your Progress")
        st.markdown("")

        page_names = list(PAGES.keys())
        current_index = list(PAGES.values()).index(st.session_state.page)
        total_steps = len(page_names)

        progress_pct = int(((current_index + 1) / total_steps) * 100)

        st.markdown(f"""
        <style>
        .circular-progress {{
            position: relative;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: conic-gradient(#987D9A {progress_pct * 3.6}deg, #EEE {progress_pct * 3.6}deg);
            margin: auto;
        }}
        .circular-progress::before {{
            content: '';
            position: absolute;
            top: 15px;
            left: 15px;
            width: 90px;
            height: 90px;
            background-color: white;
            border-radius: 50%;
        }}
        .circular-progress span {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            font-size: 18px;
        }}
        </style>

        <div class="circular-progress">
            <span>{progress_pct}%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")
        st.markdown(f"You’re currently on: **{page_names[current_index]}**")

        st.markdown("---")

        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.session_state.page = "overview"
            st.rerun()
    else:
        st.info("Please log in to access features.")
        if st.button("Log In / Sign Up"):
            st.session_state.page = "auth"
    
# -----------------------------------
#     Homepage / Feature Overview 
# -----------------------------------
if st.session_state.page == "overview":
    st.markdown("""
    <div style='text-align: justify; font-size: 20px;'>
        🧭 <strong>Campus Care</strong> is a student-centered wellbeing platform designed to support your mental health, emotional balance, and personal growth throughout your university journey. Whether you need a moment to reflect, a safe space to check in, or tools to help manage stress, Campus Care is here to walk with you — every step of the way.
    </div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("##### With Campus Care, you can:\n")
    st.markdown("")

    features = [
        {"step": "1", "title": "📊 Dashboard", "desc": "View your mental health facts, track progress, and access your toolkit."},
        {"step": "2", "title": "🧠 Self-Assessment", "desc": "Take a quick 7-question quiz to check your current well-being score."},
        {"step": "3", "title": "🚦 Smart Pathways", "desc": "Based on your results, navigate between High-Risk or Low-Risk pathways."},
        {"step": "4", "title": "🛡️ Risk Detection", "desc": "Multi-layered analysis helps identify your support needs accurately."},
        {"step": "5", "title": "💬 Cluster Insights", "desc": "Discover which group you align with based on similar behavior and wellness patterns."},
        {"step": "6", "title": "🎁 Personalized Dashboard", "desc": "Earn badges, track your progress, and see everything in one view!"}
    ]


    for i in range(0, len(features), 2):
        col1, col2 = st.columns(2)
        for j, col in enumerate([col1, col2]):
            if i + j < len(features):
                f = features[i + j]
                col.markdown(f"""
                <div class="feature-card">
                    <div class="step-circle">{f['step']}</div>
                    <h4 style="margin: 0; font-size: 1.2em;">{f['title']}</h4>
                    <p style="margin: 8px 0 0 0; font-size: 1em; color: #333;">{f['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Because caring for your mind is just as important as your grades.\n")
    st.markdown("##### Welcome to a campus that cares — welcome to Campus Care.\n")

# --------------------------------------
#     Authentication Page (DB-based) 
# --------------------------------------
elif st.session_state.page == "auth":
    st.markdown("### 👤 Welcome to UOW Wellness App")
    auth_mode = st.radio("Choose an option", ["Log In", "Sign Up"], horizontal=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if auth_mode == "Sign Up":
        if st.button("Sign Up"):
            if create_user(username, password):
                st.success("Account created. Please log in.")
            else:
                st.error("Username already exists.")
    else:
        if st.button("Log In"):
            if validate_user(username, password):
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.session_state.username = username  
                st.session_state.page = "self_check"  
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.markdown("</div>", unsafe_allow_html=True)
    

#-------------------
#   Self Check 
#-------------------
elif st.session_state.page == "self_check":
    st.markdown("""
    ## Step 1: A Day in Uni Life – Reflecting on Your Wellbeing
    
    Meet **Kai**, your friendly wellness companion throughout this journey.  
    Kai is here to guide you through a typical day in uni life — from waking up to winding down — helping you pause, reflect, and check in with yourself.
    
    Together, you’ll explore how you’ve been feeling lately, using a few simple questions based on the Short Warwick-Edinburgh Mental Wellbeing Scale (SWEMWBS).  
    There are no right or wrong answers — just be honest with yourself.
    """)

    st.markdown("") 
    
    # Likert scale legend
    st.markdown("""
    **Rate based on this scale:**
    😞 1 – None of the time | 😕 2 – Rarely | 😐 3 – Some of the time | 🙂 4 – Often | 😄 5 – All of the time
    """)
    st.markdown("---")

    questions = [
        {
            "scene": "🌅 Scene 1: Morning Routine in Dorm",
            "desc": "Kai:\n\"Another busy day ahead! But let's take a moment first… Have you been feeling optimistic about how things are going in life and studies?\""
        },
        {
            "scene": "📚 Scene 2: Rushing to Lecture",
            "desc": "Kai:\n\"Campus life is hectic — but have you felt useful or like your contributions matter, whether in class, clubs, or helping friends?\""
        },
        {
            "scene": "🖥️ Scene 3: Group Project Work",
            "desc": "Kai:\n\"When faced with choices — like how to manage your group or schedule — have you felt confident making decisions?\""
        },
        {
            "scene": "🍛 Scene 4: Cafeteria Chat with Friends",
            "desc": "Kai:\n\"Social moments can be comforting. Lately, have you felt relaxed, even in social or academic settings?\""
        },
        {
            "scene": "🏞️ Scene 5: Solo Break Between Classes",
            "desc": "Kai:\n\"Let’s be real — uni can be draining. But how have you felt about yourself overall? Have you been feeling useful?\""
        },
        {
            "scene": "🧑‍🤝‍🧑 Scene 6: Club Meeting or Society Hangout",
            "desc": "Kai:\n\"Whether with flatmates, classmates, or society friends — have you felt close or connected to others lately?\""
        },
        {
            "scene": "🌌 Scene 7: Evening Study or Journaling",
            "desc": "Kai:\n\"At the end of the day — when thinking through assignments or your own thoughts — have you felt clear-headed?\""
        }
    ]

    if "swemwbs_responses" not in st.session_state:
        st.session_state.swemwbs_responses = [3] * 7  # Default mid-score

    total_score = 0
    for i, q in enumerate(questions):
        st.markdown(f"#### {q['scene']}")
        st.markdown(q["desc"])
        response = st.slider(
            f"Q{i+1}", 1, 5, st.session_state.swemwbs_responses[i],
            format="%d", key=f"q{i+1}", label_visibility="collapsed"
        )
        st.session_state.swemwbs_responses[i] = response
        total_score += response
        st.markdown("---") 

    st.markdown("")

    # Initialize snapshot state if not yet set
    if "show_snapshot" not in st.session_state:
        st.session_state.show_snapshot = False
    
    if st.button("📊 Show My Wellbeing Snapshot"):
        st.session_state.show_snapshot = True

        # Save visit immediately when snapshot is first shown
        user_id = get_user_id(st.session_state.username)
        risk_level = "High" if total_score <= 21 else "Low"
        save_self_check_visit(user_id, total_score, risk_level)
    
    if st.session_state.show_snapshot:
        st.markdown("### 📊 Step 2: Your Wellbeing Snapshot")
        st.markdown(f"**Your Total Score:** {total_score} / 35")
    
        # Scoring explanation
        with st.expander("ℹ️ How This Works"):
            st.markdown("""
            Each question is scored from 1 to 5:
            - 😢 None of the time → 1 point  
            - 😄 All of the time → 5 points
    
            **Total possible score: 7–35**
    
            - 🔴 Score ≤ 21 → High-Risk Pathway  
            - 🟢 Score > 21 → Low-Risk Pathway
            """)    
            
        # Show result and navigation buttons
        if total_score <= 21:
            st.markdown("🔴 **High-Risk Pathway**")
            st.warning("You might benefit from additional support. Let’s explore some helpful resources together.")
            if st.button("🔍 View Supportive Resources"):
                st.session_state.page = "high_risk_pathway"
                st.rerun()
        else:
            st.markdown("🟢 **Low-Risk Pathway**")
            st.success("Great! You’re showing strong signs of wellbeing. Let’s keep the momentum going.")
            if st.button("➡️ Continue"):
                st.session_state.page = "low_risk_pathway"
                st.rerun()

# --------------------
# LOW RISK FLOW
# --------------------
elif st.session_state.page == "low_risk_pathway":
    st.title("🟢 Low-Risk Pathway")
    st.markdown("""
    Your mental well-being appears to be in a positive range!

    These short awareness modules are designed to:
    - Increase self-understanding
    - Raise awareness about emotional well-being
    - Introduce daily micro-habits for mental resilience

    Let’s go through them together — one step at a time.
    """)

    if st.button("🎓 Begin Wellness Modules"):
        st.session_state.page = "low_risk_modules"
        st.rerun()

elif st.session_state.page == "low_risk_modules":
    st.title("Wellness Micro-Modules")

    # Track module completion in session state
    if "completed_modules" not in st.session_state:
        st.session_state.completed_modules = {"mod1": False, "mod2": False, "mod3": False}

    # --- Module 1 ---
    st.markdown("---")
    st.markdown("### 🎓 Module 1: Understanding Anxiety in University Life")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📘 Overview", 
        "👩‍🎓 Meet Zara & Ethan", 
        "💬 Real Symptoms", 
        "🙈 Why We Hide It", 
        "🧠 Why This Matters"
    ])
    
    with tab1:
        st.markdown("""
        This module is designed to help you understand **what anxiety can look like** in university students.  
        While some signs are visible, many symptoms are subtle and internal.
    
        You may not even realize someone around you — or even you — are dealing with anxiety until it's overwhelming.
    
        Let's break it down together.
        """)
    
    with tab2:
        st.markdown("""
        #### 👩 Zara  
        - Always prepared, active in clubs, a top student.  
        - But secretly, she rereads her emails 10 times before sending them.  
        - She's constantly anxious about sounding “dumb” or “wrong.”  
        - Even small mistakes send her into spirals of doubt.
    
        #### 👦 Ethan  
        - The funny one in class, known for keeping spirits up.  
        - Lately, he’s been missing lectures, skipping meals, and struggling to concentrate.  
        - He laughs it off — but deep down, he feels overwhelmed and stuck.
    
        These stories reflect what many students experience — even when no one else notices.
        """)
    
    with tab3:
        st.markdown("""
        Anxiety doesn’t always scream — sometimes, it whispers through habits and feelings like:
    
        - 💤 Waking up already feeling drained  
        - 📉 Constant self-doubt about performance  
        - 🙅 Skipping hangouts due to mental fatigue  
        - 😐 Feeling “numb” or disconnected even when surrounded by friends  
        - 🔁 Replaying small events over and over in your head  
    
        These symptoms don’t mean someone is failing — it means they’re human and overwhelmed.
        """)
    
    with tab4:
        st.markdown("""
        Many students **hide their anxiety** for reasons like:
    
        - “Everyone else seems fine, so I should be too.”  
        - “I don’t want to be dramatic or get judged.”  
        - “If I speak up, people will think I’m weak.”  
        - “It’s not a big deal… right?”
    
        But here’s the truth:  
        ❗ You don’t need a breakdown to justify your feelings.  
        ❗ You don’t need to suffer silently to be strong.
    
        Real strength is acknowledging the struggle — and choosing to take care of yourself anyway.
        """)
    
    with tab5:
        st.markdown("""
        Anxiety can sneak into your academic, social, and personal life if left unchecked.
    
        - You may lose motivation  
        - You may isolate yourself  
        - You may burn out silently
    
        But identifying it early and being kind to yourself is the first step toward managing it.
    
        > 🧡 Remember: You're allowed to take up space. You're allowed to ask for help.  
        > Healing starts with honesty — and you’ve already taken the first step.
    
        Mental wellness is about more than surviving uni — it's about building a life where you feel safe, capable, and supported.
        """)

    if not st.session_state.completed_modules["mod1"]:
        st.markdown("")
        st.markdown("")
        st.markdown("💬 **Reflection:** What's something you've learned or resonated with anxiety in university life?")
        mod1_reflection = st.text_area("Your thoughts on anxiety awareness", key="mod1_input")
    
        if st.button("✅ Complete Module 1"):
            if mod1_reflection.strip():
                user_id = get_user_id(st.session_state.username)
                save_reflection(user_id, "Module 1", mod1_reflection)
            st.session_state.completed_modules["mod1"] = True
            st.rerun()

    # --- Module 2 ---
    st.markdown("---")
    if st.session_state.completed_modules["mod1"]:
        with st.container():
            st.markdown("### 🧘 Module 2: 2-Minute Gratitude Reflection")

            tab1, tab2, tab3 = st.tabs([
                "🌤️ Why Gratitude?", 
                "📝 Try This Now", 
                "🔁 Make It a Habit"
            ])
            
            with tab1:
                st.markdown("""
                When everything feels overwhelming, **gratitude can shift your brain’s focus** — even for just a moment.
            
                Gratitude isn't about ignoring your problems. It's about noticing what's *still* okay despite them.
            
                Researchers found that students who practiced gratitude reported:
            
                - Lower stress levels
                - Better sleep
                - A more positive outlook during exams
            
                It’s a small pause, but it can have a **big impact** on your emotional balance.
                """)
            
            with tab2:
                st.markdown("""
                Close your eyes (or just slow your thoughts) and reflect on these prompts:
            
                - 🍲 *What is something I ate recently that I enjoyed?*
                - 💬 *Who is one person — even if just online — who made me smile this week?*
                - 🌱 *What’s one small win I had today?*
            
                Even on hard days, these moments are still real.
            
                > “Gratitude turns what we have into enough.”  
                > – A reminder from Kai 💙
                """)
            
            with tab3:
                st.markdown("""
                Just **2 minutes a day** can help:
            
                - Ground you when you feel scattered  
                - Train your mind to notice the good  
                - Reduce the spiral of stress  
            
                🧠 **Tip:** Set a phone reminder titled:  
                _"What’s one good thing today?"_
            
                Don’t overthink it — just answer honestly.
            
                Consistency matters more than perfection.
                """)

            if not st.session_state.completed_modules["mod2"]:
                st.markdown("")
                st.markdown("")
                st.markdown("💬 **Reflection:** What is something you’re grateful for today?")
                mod2_reflection = st.text_area("Write a gratitude note", key="mod2_input")
            
                if st.button("✅ Complete Module 2"):
                    if mod2_reflection.strip():
                        user_id = get_user_id(st.session_state.username)
                        save_reflection(user_id, "Module 2", mod2_reflection)
                    st.session_state.completed_modules["mod2"] = True
                    st.rerun()

    # --- Module 3 ---
    st.markdown("---")
    if st.session_state.completed_modules["mod2"]:
        with st.container():
            st.markdown("### 💡 Module 3: What Does Mental Wellness Mean to You?")

            tab1, tab2, tab3 = st.tabs([
                "🔍 Not a One-Size Answer", 
                "🧩 Reflect On Yourself", 
                "🎯 Why This Matters"
            ])
            
            with tab1:
                st.markdown("""
                Mental wellness looks different for everyone.
            
                - For some, it’s *getting out of bed before 10am*  
                - For others, it’s *feeling safe to cry*  
                - Or simply *feeling at ease in your own mind*
            
                Don’t compare your version of wellness to others.
            
                > Your journey isn’t less valid just because it looks different.
                """)
            
            with tab2:
                st.markdown("""
                Take a quiet moment to reflect or jot these down:
            
                - 😌 *When do I feel most at peace?*
                - ⚖️ *What throws me off balance emotionally?*
                - ❤️ *What support would I ask for if I wasn’t afraid?*
                - 🛠️ *What small habit helps me reset my mind?*
            
                Mental wellness isn’t an end goal — it’s an ongoing awareness of what helps you **feel like yourself** again.
                """)
            
            with tab3:
                st.markdown("""
                Knowing *your own definition* of mental wellness helps you:
            
                - Set realistic goals  
                - Advocate for what you need  
                - Spot early warning signs  
            
                Most importantly, it reminds you that:
            
                🧡 **You’re not broken. You’re human.**  
                You deserve support that meets *your* needs — not just generic advice.
            
                This reflection is about **self-kindness**, not self-judgment.
                """)
    
            if not st.session_state.completed_modules["mod3"]:
                st.markdown("")
                st.markdown("")
                st.markdown("💬 **Reflection:** What does mental wellness mean to *you* right now?")
                mod3_reflection = st.text_area("Your mental wellness definition", key="mod3_input")
            
                if st.button("✅ Complete Module 3"):
                    if mod3_reflection.strip():
                        user_id = get_user_id(st.session_state.username)
                        save_reflection(user_id, "Module 3", mod3_reflection)
                    st.session_state.completed_modules["mod3"] = True
                    st.rerun()    

    # Final action button if all modules completed
    if all(st.session_state.completed_modules.values()):
        st.success("🎉 All modules completed! You can now proceed to your dashboard.")
        if st.button("🚀 Go to Dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()



# --------------------------
# HIGH RISK PATHWAY SECTION
# --------------------------
elif st.session_state.page == "high_risk_pathway":
    with st.container():

        model = joblib.load("log_stacking_model.pkl")
    
        st.title("🔴 High-Risk Pathway")
        st.markdown("Kai: *Thanks for continuing this journey with me. These next questions will help me understand more about what you’re going through.*")
    
        # Likert scale legend
        st.markdown("""
        **Rate based on this scale:**
        😞 1 – None of the time | 😕 2 – Rarely | 😐 3 – Some of the time | 🙂 4 – Often | 😄 5 – All of the time
        """)
        st.markdown("---")
    
        st.markdown("#### 🧑‍🎓 Scene 1: Let’s Start With You")
        with st.chat_message("assistant"):
            st.markdown("*How old are you, if you don't mind me asking?*")
        age = st.number_input("🎂 Your Age", min_value=16, max_value=30, step=1)
        st.markdown("---")
    
        st.markdown("#### 📖 Scene 2: Academic Life Check-In")
        with st.chat_message("assistant"):
            st.markdown("*Uni life can be intense! On average, how many hours a week do you spend studying?*")
        study_hours = st.number_input("📘 Study Hours Per Week", min_value=0, max_value=100, step=1)
        st.markdown("---")
    
        with st.chat_message("assistant"):
            st.markdown("*If you had to describe your academic workload, what would it be?*")
        academic_workload = st.slider("📈 Academic Workload", 1, 5, 3, format="%d")
        st.markdown("---")
    
        with st.chat_message("assistant"):
            st.markdown("*Do you often feel pressured by your coursework?*")
        coursework_pressure = st.slider("📝 Coursework Pressure", 1, 5, 3, format="%d")
        st.markdown("---")
    
        st.markdown("#### 💰 Scene 3: Finances & You")
        with st.chat_message("assistant"):
            st.markdown("*Do money issues often add to your stress?*")
        financial_stress = st.slider("💵 Financial Stress", 1, 5, 3, format="%d")
        st.markdown("---")
    
        st.markdown("#### 💤 Scene 4: Sleep Habits")
        with st.chat_message("assistant"):
            st.markdown("*Let’s talk sleep. On average, how many hours do you get each night?*")
        sleep_hours = st.number_input("🌙 Sleep Hours Per Night", min_value=0.0, max_value=12.0, step=0.5)
        st.markdown("---")
    
        st.markdown("#### 🏃 Scene 5: Staying Active")
        with st.chat_message("assistant"):
            st.markdown("*How often do you get moving — like exercising, walking, or stretching?*")
        physical_activity = st.slider("🏋️‍♀️ Physical Activity Frequency", 1, 5, 3, format="%d")
        st.markdown("---")
    
        st.markdown("#### 👥 Scene 6: Social Life")
        with st.chat_message("assistant"):
            st.markdown("*Are you involved in clubs, societies, or volunteering?*")
        cocurricular = st.slider("🎭 Co-Curricular Involvement", 1, 5, 3, format="%d")
        st.markdown("---")
    
        with st.chat_message("assistant"):
            st.markdown("*Do you often feel isolated or disconnected from your peers?*")
        isolation = st.slider("🕳️ Isolation Frequency", 1, 5, 3, format="%d")
        st.markdown("---")
    
        st.markdown("#### 🚨 Scene 7: Mental Health Moments")
        with st.chat_message("assistant"):
            st.markdown("*In the past 2 weeks, have you had any thoughts of hurting yourself?*")
        suicidal_thoughts = st.radio("💭 Recent Suicidal Thoughts", ["No", "Yes"])
        suicidal_binary = 1 if suicidal_thoughts == "Yes" else 0
    
        st.markdown("---")

        # Ensure correct order
        input_dict = {
            "Age": age,
            "Study_Hours_Per_Week": study_hours,
            "Academic_Workload": academic_workload,
            "Coursework_Pressure": coursework_pressure,
            "Sleep_Hours_Per_Night": sleep_hours,
            "Physical_Activity_Freq": physical_activity,
            "Financial_Stress": financial_stress,
            "CoCurricular_Involvement": cocurricular,
            "Isolation_Frequency": isolation,
            "Recent_Suicidal_Thoughts": suicidal_binary
        }
        
        # Convert to dataframe with correct column order
        input_df = pd.DataFrame([input_dict])[[
            "Age",
            "Study_Hours_Per_Week",
            "Academic_Workload",
            "Coursework_Pressure",
            "Sleep_Hours_Per_Night",
            "Physical_Activity_Freq",
            "Financial_Stress",
            "CoCurricular_Involvement",
            "Isolation_Frequency",
            "Recent_Suicidal_Thoughts"
        ]]

        if st.button("🔎 Analyze My Mental Risk Level"):
            try:
                prediction = model.predict(input_df)[0]
    
                st.markdown("## 📊 Kai’s Check-In Result")
                if prediction == 0:
                    st.success("🟢 Minimal to Mild Risk\nKai: *You're showing early signs, but you're managing well. Keep checking in with yourself!*")
                    st.info("Redirecting you to the Low-Risk Wellness Pathway for encouragement and growth tips.")
                    time.sleep(5)
                    st.session_state.page = "low_risk_pathway"
                    st.rerun()
                elif prediction == 1:
                    st.warning("🟠 Moderate Risk\nKai: *There are some warning signs. You might benefit from support circles or peer check-ins.*")
                elif prediction == 2:
                    st.error("🔴 Severe Risk\nKai: *I'm concerned about your well-being. Please know that you're not alone. Let’s explore support options together.*")

                if prediction in [1, 2]:
                    # Prepare unnormalized vector for cluster assignment
                    user_vector = [
                        coursework_pressure, study_hours, academic_workload,
                        cocurricular, isolation, physical_activity,
                        sleep_hours, suicidal_binary, financial_stress, age
                    ]
                    
                    # Determine group label
                    group_label = "Moderate" if prediction == 1 else "Severe"
                    
                    # Assign to nearest cluster
                    cluster_assignment = assign_cluster(user_vector, group_label)

                    cluster_profiles = pd.read_csv("all_cluster_profiles.csv")
                    cluster_data = cluster_profiles[
                        (cluster_profiles["Cluster"] == cluster_assignment) & (cluster_profiles["Group"] == group_label)
                    ]

                    # Extract cluster name (make sure it's in your CSV)
                    cluster_name = cluster_data["Cluster_Name"].values[0]

                    st.info(f"📌 Assigned to Cluster: {cluster_assignment} ({group_label})")
    
                    # ----------------------------
                    #    RADAR CHART + INSIGHTS 
                    # ----------------------------
                    radar_features = [
                        "Coursework_Pressure", "Study_Hours_Per_Week", "Academic_Workload", "CoCurricular_Involvement",
                        "Isolation_Frequency", "Physical_Activity_Freq", "Sleep_Hours_Per_Night",
                        "Recent_Suicidal_Thoughts", "Financial_Stress", "Age"
                    ]
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=cluster_data[radar_features].values.flatten(),
                        theta=radar_features,
                        fill='toself',
                        name=f"Cluster {cluster_assignment}"
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                        title=f"🧭 Profile Radar: Cluster {cluster_name} ({group_label})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Profile Insights
                    with st.expander("🧠 View Cluster Insights"):
                        if group_label == "Moderate":
                            if cluster_assignment == 0:
                                st.markdown("""
                                ### 🌀 *The Overwhelmed Balancer* – Moderate Group
                                
                                **Profile Insights:**
                                - High coursework pressure (**0.71**) despite moderate academic workload (**0.54**) and low study hours (**0.23**) — possibly due to procrastination or poor stress coping.
                                - Moderate levels of **co-curricular involvement (0.40)** and **physical activity (0.47)** — trying to stay balanced.
                                - Noticeable **financial stress (0.53)** and **early signs of suicidal thoughts (0.32)** — suggesting emotional vulnerability.
                                - Average age is around **20–21**, possibly facing academic transition stress.
                            
                                **Summary:**  
                                This group may struggle with time management and emotional stress despite a manageable workload.
                            
                                **Advice Focus:**
                                - Encourage **time management**, **stress reappraisal**, and **building coping strategies**.
                                - Introduce reflective journaling and resilience-focused exercises.
                                """)
                    
                            elif cluster_assignment == 1:
                                st.markdown("""
                                ### 🌫️ *The Drifting Observer* – Moderate Group
                                
                                **Profile Insights:**
                                - Lowest **co-curricular involvement (0.22)** and **physical activity (0.42)** — indicating social and physical disengagement.
                                - Younger age group (**~18 years**) with **low coursework pressure (0.44)** and **manageable stress levels**.
                                - Mild presence of suicidal thoughts (**0.33**) suggests potential early warning signs.
                                - Minimal isolation (**0.25**) — students are not disconnected, but may feel **unmotivated**.
                            
                                **Summary:**  
                                These students are socially and physically inactive, possibly due to emotional detachment or a lack of academic direction.
                            
                                **Advice Focus:**
                                - Encourage **peer bonding**, **structured academic goals**, and **gentle motivational interventions**.
                                - Promote participation in low-pressure communities and interest-based groups.
                                """)
                                
                        elif group_label == "Severe":
                            if cluster_assignment == 1:
                                st.markdown("""
                                ### 🎯 *The Silent Perfectionist* – Severe Group
                                
                                **Profile Insights:**
                                - Extremely high coursework pressure (**0.78**) despite only moderate academic workload (**0.62**) — suggests **internalized pressure or perfectionism**.
                                - **Less financial stress** — their stress likely comes from **self-imposed expectations**, not external hardship.
                                - Slightly **better sleep quality (0.56)** and **lower suicidal thoughts (0.33)** than other severe groups — but signs may be **masked**.
                                - Moderate levels of **co-curricular involvement** and **physical activity** — indicating **social participation**, but emotional weight remains.
                            
                                **Summary:**  
                                This group is high-functioning on the outside but battles **internal perfectionism** that silently impacts mental health.
                            
                                **Advice Focus:**
                                - Address **maladaptive perfectionism** and **unhealthy self-expectations**.
                                - Promote **healthy goal-setting**, **self-compassion**, and **emotional self-awareness**.
                                """)
    
                            elif cluster_assignment == 0:
                                st.markdown("""
                                ### 💢 *The Struggling Achiever* – Severe Group
                                
                                **Profile Insights:**
                                - High **academic workload (0.68)** and **coursework pressure (0.69)** — academic overload is intense.
                                - Very low **sleep (0.47)** and **extremely high financial stress (1.00)** — signs of major life strain.
                                - **High suicidal thoughts (0.32)** despite some participation in physical and co-curricular activities — may be masking severe distress.
                                - Very young age (**~17–18 years old**) suggests difficulty adjusting to university-level challenges.
                            
                                **Summary:**  
                                These students are under severe academic, financial, and emotional stress and may be silently struggling.
                            
                                **Advice Focus:**
                                - Provide **crisis support**, **emergency counseling**, and **financial aid pathways**.
                                - Promote early intervention through trained peer listeners and hotline accessibility.
                                """)
                    
                    # 💾 Save to database
                    user_id = get_user_id(st.session_state.username)  # Make sure username is stored in session_state
                    if user_id:
                        save_high_risk_response(
                            user_id, age, study_hours, coursework_pressure, academic_workload,
                            sleep_hours, physical_activity, isolation, financial_stress,
                            cocurricular, suicidal_binary, prediction, cluster_assignment
                        )

                        st.success("🎉 Thanks for opening up, even when things are hard. You’re not alone in this — and support is just a click away. Together, we can start creating a healthier space for you.")
                        
                    else:
                        st.warning("⚠️ Could not save response. User not found.")
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown("")
        st.markdown("---")
        if st.button("🚀 Go to Dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()

# -----------------
#    Dashboard    
# -----------------
elif st.session_state.page == "dashboard":
    st.title("📊 Your Mental Wellness Dashboard")

    user_id = get_user_id(st.session_state.username)

    # Show Visit Summary
    total, low, high = get_self_check_stats(user_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>Total Check-Ins</h3>
                <p style='font-size: 24px; font-weight: bold;'>{total}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>🟢 Low-Risk Sessions</h3>
                <p style='font-size: 24px; font-weight: bold;'>{low}</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>🔴 High-Risk Sessions</h3>
                <p style='font-size: 24px; font-weight: bold;'>{high}</p>
            </div>
        """, unsafe_allow_html=True)

    # Two-column layout
    left_col, right_col = st.columns(2)

    # 🔬 Left: Most Recent Cluster Visits
    with left_col:
        st.markdown("### 🔬 Recent High-Risk Clusters")
        recent_clusters = get_recent_clusters(user_id)
        if recent_clusters:
            for cluster in recent_clusters:
                st.markdown(f"""
                    <div class="cluster-card">
                        <h4>{cluster['label']}</h4>                     
                        <p><strong>Description:</strong> {cluster['description']}</p>
                        <p><strong>Date:</strong> {cluster['date']}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("All check-ins are low risk!")

    # 🏅 Right: Award Badge
    with right_col:
        st.markdown("### 🏅 Badge")
        
        if total == 0:
            st.info("Complete a check-in to start earning badges!")
        else:
            low_ratio = low / total
            high_ratio = high / total
    
            # Stress-Free Champ 
            if low == total:
                st.markdown("""
                <style>
                .container {
                  display: flex;
                  flex-direction: column;
                  align-items: center;
                  justify-content: center;
                  font-size: 2em;
                  font-weight: 900;
                  color: #e10600;
                  position: relative;
                  transition: all 1s ease;
                  text-align: center;
                  margin: 50px auto;
                }
            
                .container__star {
                  transition: all 0.7s ease-in-out;
                }
            
                .first {
                  position: absolute;
                  top: 20px;
                  left: 50px;
                  transition: all 0.7s ease-in-out;
                }
            
                .svg-icon {
                  position: absolute;
                  fill: #e94822;
                  z-index: 1;
                }
            
                .star-eight {
                  background: #efd510;
                  width: 150px;
                  height: 150px;
                  position: relative;
                  text-align: center;
                  animation: rot 3s infinite;
                  border-radius: 16px;
                }
            
                .star-eight::before {
                  content: '';
                  position: absolute;
                  top: 0;
                  left: 0;
                  height: 150px;
                  width: 150px;
                  background: #efd510;
                  transform: rotate(135deg);
                  border-radius: 16px;
                }
            
                .container:hover .container__star {
                  transform: rotateX(70deg) translateY(250px);
                  box-shadow: 0px 0px 120px -100px #e4e727;
                }
            
                .container:hover .svg-icon {
                  animation: grow 1s linear infinite;
                }
            
                @keyframes rot {
                  0%   { transform: rotate(0deg); }
                  50%  { transform: rotate(340deg); }
                  100% { transform: rotate(0deg); }
                }
            
                @keyframes grow {
                  0%   { transform: rotate(0deg); }
                  25%  { transform: rotate(-5deg); }
                  75%  { transform: rotate(5deg); }
                  100% { transform: scale(1) rotate(0deg); }
                }
                </style>
            
                <div class="container">
                  <svg class="svg-icon" height="100" width="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                    <path d="M62.11,53.93c22.582-3.125,22.304-23.471,18.152-29.929-4.166-6.444-10.36-2.153-10.36-2.153v-4.166H30.099v4.166s-6.194-4.291-10.36,2.153c-4.152,6.458-4.43,26.804,18.152,29.929l5.236,7.777v8.249s-.944,4.597-4.833,4.986c-3.903,.389-7.791,4.028-7.791,7.374h38.997c0-3.347-3.889-6.986-7.791-7.374-3.889-.389-4.833-4.986-4.833-4.986v-8.249l5.236-7.777Zm7.388-24.818s2.833-3.097,5.111-1.347c2.292,1.75,2.292,15.86-8.999,18.138l3.889-16.791Zm-44.108-1.347c2.278-1.75,5.111,1.347,5.111,1.347l3.889,16.791c-11.291-2.278-11.291-16.388-8.999-18.138Z">
                    </path>
                  </svg>  
            
                  <div class="container__star">
                    <div class="star-eight"></div>
                  </div>
                </div>
                <div style="text-align:center; font-family: 'Segoe UI', sans-serif; font-size: 1.5rem; margin-top: 20px; color: #444;">
                  🏅 Stress-Free Champ
                </div>
                <div style="text-align:center; font-family: 'Segoe UI', sans-serif; font-size: 1rem; margin-top: 10px; color: #333;">
                  This badge celebrates your calm and steady mindset. Keep it up, you're doing great!
                </div>
                """, unsafe_allow_html=True)
    
            elif high == total:
                st.markdown("""
                <style>
                body {
                  font-family: 'Allerta Stencil', sans-serif;
                  padding: 0;
                  margin: 0;
                  background: #efefef;
                }
            
                .badge-wrapper {
                  display: flex;
                  flex-direction: column;
                  align-items: center;
                  justify-content: flex-start;
                  padding-top: 80px;
                  padding-bottom: 30px;
                }
            
                .badge {
                  position: relative;
                  letter-spacing: 0.08em;
                  color: #fff;
                  display: flex;
                  justify-content: center;
                  align-items: center;
                  text-decoration: none;
                  transition: transform 0.3s ease;
                  transform: rotate(-14deg);
                  text-align: center;
                  filter: drop-shadow(0.25em 0.7em 0.7em rgba(0,0,0, 0.6));
                  font-size: calc(11px + (14 * ((100vw - 420px) / 860)));
                }
            
                @media screen and (max-width: 420px) {
                  .badge {
                    font-size: 11px;
                  }
                }
            
                @media screen and (min-width: 1280px) {
                  .badge {
                    font-size: 25px;
                  }
                }
            
                .badge::before {
                  content: "";
                  position: absolute;
                  top: 50%;
                  left: 50%;
                  transform: translate(-50%, -50%);
                  display: block;
                  width: 10em;
                  height: 10em;
                  border-radius: 100%;
                  background: #FF9D23;
                  opacity: 0.8;
                  transition: opacity 0.3s linear;
                }
            
                .badge:hover {
                  color: #fff;
                  text-decoration: none;
                  transform: rotate(-10deg) scale(1.05);
                }
            
                .badge:hover::before {
                  opacity: 0.9;
                }
            
                .badge svg {
                  position: absolute;
                  top: 50%;
                  left: 50%;
                  transform: translate(-50%, -50%);
                  display: block;
                  z-index: 0;
                  width: 10em;
                  height: 10em;
                }
            
                .badge span {
                  display: block;
                  background: #FADA7A;
                  border-radius: 0.4em;
                  padding: 0.4em 1em;
                  z-index: 1;
                  min-width: 11em;
                  border: 1px solid;
                  text-transform: uppercase;
                }
                </style>
            
                <div class="badge-wrapper">
                  <a href="#" class="badge">
                    <svg viewBox="0 0 210 210">
                      <g stroke="none" fill="none">
                        <path d="M22,104.5 C22,58.9365081 58.9365081,22 104.5,22 C150.063492,22 187,58.9365081 187,104.5" id="top"></path>
                        <path d="M22,104.5 C22,150.063492 58.9365081,187 104.5,187 C150.063492,187 187,150.063492 187,104.5" id="bottom"></path>
                      </g>
                      <circle cx="105" cy="105" r="62" stroke="currentColor" stroke-width="1" fill="none" />
                      <text width="200" font-size="20" fill="currentColor">
                        <textPath startOffset="50%" text-anchor="middle" alignment-baseline="middle" xlink:href="#top">
                          Stay Aware
                        </textPath>
                      </text>
                      <text width="200" font-size="20" fill="currentColor">
                        <textPath startOffset="50%" text-anchor="middle" alignment-baseline="middle" xlink:href="#bottom">
                          Mind Your Stress
                        </textPath>
                      </text>
                    </svg>
                    <span>🚨 Risk Alert Explorer</span>
                  </a>
            
                  <div style="text-align:center; font-family: 'Segoe UI', sans-serif; font-size: 1.1rem; margin-top: 20px; color: #333;">
                    <br><br><br><br>
                    This badge is awarded when all your recent check-ins indicate high stress.  
                    You're showing awareness by consistently checking in during tough moments — that's a powerful first step toward resilience
                  </div>
                </div>
                """, unsafe_allow_html=True)
    
            # Balanced tracker  
            elif abs(low - high) <= 1 and total >= 2:
                    st.markdown("""
                    <style>
                    /* Embed Ubuntu Font */
                    @import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;700&display=swap');
                
                    .banff {
                        height: 320px;
                        width: 350px;
                        margin: 20px 80px 10px 170px;
                        font-family: 'Ubuntu', sans-serif;
                    }
                
                    .banff-border {
                        align-items: center;
                        position: relative;
                        background: none;
                        background: #cddce0;
                        height: 300px;
                        width: 215px;
                        overflow: hidden;
                    }
                
                    .banff-border:before {
                        content: "";
                        position: relative;
                        background: #cddce0;
                        border-radius: 0 60px 60px 0;
                        height: 400px;
                        width: 300px;
                        z-index: -2;
                    }
                
                    .banff-frame {
                        position: absolute;
                        background: none;
                        height: 330px;
                        width: 250px;
                        border: 19px solid #334d63;
                        border-radius: 60px;
                        margin: -20px 0 0 -20px;
                        overflow: hidden;
                        z-index: 5;
                    }
                
                    .banff-header {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        background: #eeede7;
                        height: 80px;
                        width: 300px;
                    }
                
                    .banff-header h2 {
                        font-size: 1.8em;
                        font-weight: 700;
                        justify-content: center;
                        align-items: center;
                        letter-spacing: 3px;
                        color: #334d63;
                        padding: 25px;
                        margin: 0;
                    }
                
                    .banff-sun {
                        position: relative;
                        background: #fcce5b;
                        height: 150px;
                        width: 150px;
                        border-radius: 50%;
                        border: 20px solid #fce39f;
                        margin: 36px auto 0 auto;
                        animation: glow 1.5s infinite;
                    }
                
                    .banff-sun:before {
                        content: "";
                        position: absolute;
                        background: #ffc107;
                        height: 100px;
                        width: 100px;
                        border-radius: 50%;
                        margin: 25px 0 0 25px;
                        animation: glow 1.5s infinite;
                    }
                
                    .banff-mountains {
                        position: relative;
                        animation: slideup 1.2s ease;
                    }
                
                    .banff-mountain-left, .banff-mountain-right {
                        position: relative;
                        width: 0;
                        height: 0;
                        border-bottom: 270px solid #334d63;
                        border-left: 170px solid transparent;
                        border-right: 170px solid transparent;
                    }
                
                    .banff-mountain-left {
                        margin: -74px 0 10px -40px;
                    }
                
                    .banff-mountain-right {
                        margin: -280px 0 0 20px;
                    }
                
                    .banff-snow-left, .banff-snow-right {
                        position: relative;
                        height: 0;
                        width: 0;
                        border-bottom: 230px solid #eeede7;
                        border-left: 170px solid transparent;
                        border-right: 160px solid transparent;
                        top: 40px;
                        margin-left: -160px;
                        z-index: 4;
                    }
                
                    .banff-country {
                        text-align: center;
                        margin: 0 auto 0 auto;
                    }
                
                    .banff-country p {
                        font-size: 1em;
                        font-weight: 500;
                        letter-spacing: 0.5px;
                        color: #334d63;
                        line-height: 1.5;
                        margin: 0;
                    }

                    @-webkit-keyframes slideup {
                    	from { 
                    		margin-top: 400px; 
                    	}
                    	
                    	to { 
                    		margin-top: 0px; 
                    	}
                    }
                    
                    @-moz-keyframes slideup {
                    	from { 
                    		margin-top: 400px; 
                    	}
                    	
                    	to { 
                    		margin-top: 0px; 
                    	}
                    }
                    
                    @keyframes slideup {
                    	from { 
                    		margin-top: 400px; 
                    	}
                    	
                    	to { 
                    		margin-top: 0px; 
                    	}
                    }
                    
                    @-webkit-keyframes glow { 
                        0% { transform: scale(0.4); } 
                        50% { transform: scale(1.1); } 
                        100% { transform: scale(1); } 
                    }
                    
                    @-moz-keyframes glow { 
                        0% { transform: scale(0.4); } 
                        50% { transform: scale(1.1); } 
                        100% { transform: scale(1); } 
                    }
                    
                    @keyframes glow { 
                        0% { transform: scale(0.4); } 
                        50% { transform: scale(1.1); } 
                        100% { transform: scale(1); } 
                    }
                    </style>
                
                    <div class="banff">
                        <div class="banff-frame"></div>
                        <div class="banff-border">
                            <div class="banff-header">
                                <h2>BALANCED TRACKER</h2>
                            </div>
                            <div class="banff-sun"></div>
                            <div class="banff-mountains">
                                <div class="banff-mountain-left">
                                    <div class="banff-snow-left"></div>
                                </div>
                                <div class="banff-mountain-right">
                                    <div class="banff-snow-right"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="banff-country">
                        <p>You’ve maintained a near-equal mix of high and low-risk check-ins. <br> Keep it steady!</p>
                    </div>
                    """, unsafe_allow_html=True)

    
            # Calm responder 
            elif low_ratio >= 0.7:
                st.markdown("""
                <style>
                body {
                  background: #8069a1;
                  padding-top: 60px;
                }
                
                svg {
                  margin: auto;
                  display: block;
                }
                
                .badge * {
                  transform-origin: 50% 50%;
                }
                
                /* Default state without animation */
                .outer, .inner, .inline, .star, .star circle {
                  transform: scale(1);
                  opacity: 1;
                }
                
                /* Animations only when hovered */
                .badge:hover .outer,
                .badge:hover .inner,
                .badge:hover .inline {
                  animation: grow 1s ease-out;
                }
                
                .badge:hover .star {
                  animation: turn 1.1s ease-out;
                }
                
                .badge:hover .star circle {
                  animation: pulse 0.7s ease-in-out;
                }
                
                .badge:hover .star circle:nth-of-type(2) { animation-delay: 0.1s; }
                .badge:hover .star circle:nth-of-type(3) { animation-delay: 0.3s; }
                .badge:hover .star circle:nth-of-type(4) { animation-delay: 0.5s; }
                .badge:hover .star circle:nth-of-type(5) { animation-delay: 0.9s; }
                
                @keyframes grow {
                  0%   { transform: scale(0); }
                  30%  { transform: scale(1.1); }
                  60%  { transform: scale(0.9); }
                  100% { transform: scale(1); }
                }
                
                @keyframes turn {
                  0%   { transform: rotate(0) scale(0); opacity: 0; }
                  60%  { transform: rotate(375deg) scale(1.1); }
                  80%  { transform: rotate(355deg) scale(0.9); }
                  100% { transform: rotate(360deg) scale(1); }
                }
                
                @keyframes pulse {
                  50% { transform: scale(1.4); }
                }
                </style>
                
                <svg class="badge" xmlns="http://www.w3.org/2000/svg" height="250" width="250" viewBox="-40 -40 400 440">
                  <circle class="outer" fill="#F9D535" stroke="#fff" stroke-width="8" stroke-linecap="round" cx="180" cy="180" r="157"/>
                  <circle class="inner" fill="#DFB828" stroke="#fff" stroke-width="8" cx="180" cy="180" r="108.3"/>
                  <path class="inline" d="M89.4 276.7c-26-24.2-42.2-58.8-42.2-97.1 0-22.6 5.6-43.8 15.5-62.4m234.7.1c9.9 18.6 15.4 39.7 15.4 62.2 0 38.3-16.2 72.8-42.1 97" stroke="#CAA61F" stroke-width="7" stroke-linecap="round" fill="none"/>
                  <g class="star">
                    <path fill="#F9D535" stroke="#fff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" d="M180 107.8l16.9 52.1h54.8l-44.3 32.2 16.9 52.1-44.3-32.2-44.3 32.2 16.9-52.1-44.3-32.2h54.8z"/>
                    <circle fill="#DFB828" stroke="#fff" stroke-width="4" cx="180" cy="107.8" r="4.4"/>
                    <circle fill="#DFB828" stroke="#fff" stroke-width="4" cx="223.7" cy="244.2" r="4.4"/>
                    <circle fill="#DFB828" stroke="#fff" stroke-width="4" cx="135.5" cy="244.2" r="4.4"/>
                    <circle fill="#DFB828" stroke="#fff" stroke-width="4" cx="108.3" cy="160.4" r="4.4"/>
                    <circle fill="#DFB828" stroke="#fff" stroke-width="4" cx="251.7" cy="160.4" r="4.4"/>
                  </g>
                </svg>
                
                <div style="text-align:center; font-size: 24px; font-weight: bold; color: black; margin-top: 10px;">
                  🌿 Calm Responder
                </div>
                
                <div style="text-align:center; font-family: 'Segoe UI', sans-serif; font-size: 16px; margin-top: 10px; color: #222;">
                  You consistently manage your stress levels with calm and clarity. Your composed responses set a great example — keep nurturing that inner peace!
                </div>
                """, unsafe_allow_html=True)

    
            # High risk watcher
            elif high_ratio >= 0.7:
                st.markdown("""
                <style>
                body {
                  background: rgba(182, 161, 122, 0.5);
                  margin: 0;
                  padding: 0;
                }
            
                .big-basin {
                  position: relative;
                  height: 350px;
                  width: 250px;
                  background: rgba(46, 63, 44, 1);
                  border: 14px solid #b6a17a;
                  border-radius: 10px 10px 140px 140px;
                  margin: 40px auto 0 auto;
                }
            
                .trees-container {
                  position: absolute;
                  background: #334d63;
                  height: 240px;
                  width: 250px;
                  overflow: hidden;
                }
            
                .tree-small, .tree-large {
                  position: absolute;
                  animation: slideup 1.5s;
                }
            
                .tree-small { margin: 70px 0 0 40px; }
                .tree-large { margin: 50px 0 0 110px; }
            
                .tree-left, .tree-right {
                  position: relative;
                  width: 0;
                }
            
                .tree-small .tree-left {
                  border-bottom: 45px solid rgba(46, 63, 44, 1);
                  border-left: 30px solid transparent;
                }
            
                .tree-small .tree-left:before {
                  content: "";
                  position: absolute;
                  border-bottom: 60px solid rgba(46, 63, 44, 1);
                  border-left: 45px solid transparent;
                  margin: 20px 0 0 -45px;
                }
            
                .tree-small .tree-left:after {
                  content: "";
                  position: absolute;
                  border-bottom: 80px solid rgba(46, 63, 44, 1);
                  border-left: 52px solid transparent;
                  margin: 45px 0 0 -52px;
                }
            
                .tree-small .tree-right {
                  border-bottom: 45px solid rgba(22, 38, 21, 1);
                  border-right: 30px solid transparent;
                  margin: -45px 0 0 30px;
                }
            
                .tree-small .tree-right:before {
                  content: "";
                  position: absolute;
                  border-bottom: 60px solid rgba(22, 38, 21, 1);
                  border-right: 45px solid transparent;
                  margin: 20px 0 0 0;
                }
            
                .tree-small .tree-right:after {
                  content: "";
                  position: absolute;
                  border-bottom: 80px solid rgba(22, 38, 21, 1);
                  border-right: 52px solid transparent;
                  margin: 45px 0 0 0;
                }
            
                .tree-small .stump {
                  background: #433825;
                  height: 40px;
                  width: 8px;
                  margin: 85px 0 0 22px;
                  position: relative;
                }
            
                .tree-small .stump:after {
                  content: "";
                  position: absolute;
                  background: #322917;
                  height: 40px;
                  width: 8px;
                  left: 8px;
                }
            
                .tree-large .tree-left {
                  border-bottom: 60px solid rgba(46, 63, 44, 1);
                  border-left: 45px solid transparent;
                }
            
                .tree-large .tree-left:before {
                  content: "";
                  position: absolute;
                  border-bottom: 75px solid rgba(46, 63, 44, 1);
                  border-left: 52px solid transparent;
                  margin: 20px 0 0 -52px;
                }
            
                .tree-large .tree-left:after {
                  content: "";
                  position: absolute;
                  border-bottom: 90px solid rgba(46, 63, 44, 1);
                  border-left: 60px solid transparent;
                  margin: 50px 0 0 -60px;
                }
            
                .tree-large .tree-right {
                  border-bottom: 60px solid rgba(22, 38, 21, 1);
                  border-right: 45px solid transparent;
                  margin: -60px 0 0 45px;
                }
            
                .tree-large .tree-right:before {
                  content: "";
                  position: absolute;
                  border-bottom: 75px solid rgba(22, 38, 21, 1);
                  border-right: 52px solid transparent;
                  margin: 20px 0 0 0;
                }
            
                .tree-large .tree-right:after {
                  content: "";
                  position: absolute;
                  border-bottom: 90px solid rgba(22, 38, 21, 1);
                  border-right: 60px solid transparent;
                  margin: 50px 0 0 0;
                }
            
                .tree-large .stump {
                  background: #433825;
                  height: 60px;
                  width: 12px;
                  margin: 90px 0 0 33px;
                  position: relative;
                }
            
                .tree-large .stump:after {
                  content: "";
                  position: absolute;
                  background: #322917;
                  height: 60px;
                  width: 12px;
                  left: 12px;
                }
            
                .banner {
                  position: relative;
                  margin: 230px 0 0 -10px;
                  height: 60px;
                  width: 270px;
                  background: #b6a17a;
                  border-radius: 0 0 16px 16px;
                  text-align: center;
                }
            
                .banner h1 {
                  position: relative;
                  top: 10px;
                  font-size: 1.5em;
                  font-weight: 700;
                  text-transform: uppercase;
                  color: #433825;
                  font-family: 'Ubuntu', sans-serif;
                  animation: fadein 2s ease-in;
                }
            
                .star {
                  position: absolute;
                  background: #d8dfe6;
                  border-radius: 50%;
                  animation: glow 1s infinite alternate;
                }
            
                .medium { height: 8px; width: 8px; }
                .small { height: 4px; width: 4px; }
            
                .one { margin: -230px 0 0 20px; }
                .two { margin: -290px 0 0 70px; }
                .three { margin: -260px 0 0 130px; }
                .four { margin: -280px 0 0 200px; }
                .five { margin: -170px 0 0 16px; }
                .six { margin: -270px 0 0 30px; }
                .seven { margin: -240px 0 0 90px; }
                .eight { margin: -288px 0 0 120px; }
                .nine { margin: -290px 0 0 170px; }
                .ten { margin: -230px 0 0 220px; }
            
                @keyframes glow {
                  0% { box-shadow: 0 0 0 0 #fff; }
                  100% { box-shadow: 0 0 4px 4px #fff; }
                }
            
                @keyframes slideup {
                  0% { margin-top: 400px; }
                  100% { margin-top: 70px; }
                }
            
                @keyframes fadein {
                  0% { opacity: 0; }
                  100% { opacity: 1; }
                }
                </style>
            
                <div class="big-basin">
                  <div class="trees-container">
                    <div class="tree-small">
                      <div class="tree-left"></div>
                      <div class="tree-right"></div>
                      <div class="stump"></div>
                    </div>
                    <div class="tree-large">
                      <div class="tree-left"></div>
                      <div class="tree-right"></div>
                      <div class="stump"></div>
                    </div>
                  </div>
                  <div class="banner">
                    <h1>High Risk Watcher</h1>
                  </div>
                  <div class="star medium one"></div>
                  <div class="star medium two"></div>
                  <div class="star medium three"></div>
                  <div class="star medium four"></div>
                  <div class="star small five"></div>
                  <div class="star small six"></div>
                  <div class="star small seven"></div>
                  <div class="star small eight"></div>
                  <div class="star small nine"></div>
                  <div class="star small ten"></div>
                </div>
                <div style="text-align:center; font-family: 'Segoe UI', sans-serif; font-size: 0.9rem; margin-top: 10px; color: #333;">
                  You've reported mostly high stress responses. <br>
                  Keep monitoring your triggers and remember, nature and reflection can be your best reset.
                </div>
                """, unsafe_allow_html=True)

    
            # Mixed journey explorer 
            else:
                st.markdown("""
                <style>
                .contributor-badge {
                  width: 250px;
                  background: linear-gradient(135deg, #BA68C8, #9575CD);
                  color: white;
                  font-weight: bold;
                  padding: 20px;
                  text-align: center;
                  clip-path: polygon(0 0, 100% 10%, 100% 100%, 0 90%);
                  font-size: 16px;
                  margin: 20px auto;
                  animation: slideIn 1s ease-in-out;
                }
    
                @keyframes slideIn {
                  from { transform: translateX(-20px); opacity: 0; }
                  to { transform: translateX(0); opacity: 1; }
                }
                </style>
    
                <div class="contributor-badge">
                🔄 Mixed Journey Explorer
                </div>
                """, unsafe_allow_html=True)

    # 📝 Display Reflections
    with st.container():
        st.markdown("""
        <div style='background-color: #EEF1FF; padding: 20px; border-radius: 12px; margin-top: 20px;'>
            <h3 style='margin-top: 0;'>📝 Reflections from Wellness Modules</h3>
        """, unsafe_allow_html=True)
        st.markdown("")

        conn = get_db_connection()
        with conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("""
                    SELECT ur.module_name, ur.reflection, ur.created_at
                    FROM user_reflections ur
                    INNER JOIN (
                        SELECT module_name, MAX(created_at) AS latest
                        FROM user_reflections
                        WHERE user_id = %s
                        GROUP BY module_name
                    ) latest_reflections
                    ON ur.module_name = latest_reflections.module_name
                    AND ur.created_at = latest_reflections.latest
                    WHERE ur.user_id = %s
                    ORDER BY ur.module_name
                """, (user_id, user_id))
                user_reflections = cursor.fetchall()
            
        # Module to question mapping
        module_questions = {
            "Module 1": "What's something you've learned or resonated with anxiety in university life?",
            "Module 2": "What is something you’re grateful today?",
            "Module 3": "What does mental wellness mean to you right now?"
        }
        
        if user_reflections:
            for ur in user_reflections:
                question = module_questions.get(ur["module_name"], "Reflection Question")
                st.markdown(f"""
                    <div style='margin-bottom: 15px; padding: 25px 15px 10px 25px; background-color: #FFF9F9; border-radius: 8px; border: 1px solid #ddd;'>
                        <p style='margin-bottom: 5px;'><strong>📌 {question}</strong></p>
                        <p style='margin: 0 0 15px 0; padding-left: 10px; border-left: 3px solid #8069a1; color: #333;'>{ur["reflection"]}</p>
                        <p style='font-size: 0.8em; color: gray; margin-top: 5px;'>📅 Submitted on: {ur["created_at"].strftime("%Y-%m-%d")}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<em>You haven’t submitted any reflections yet.</em>", unsafe_allow_html=True)
    
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #EEEEEE;
            color: #444;
            text-align: center;
            font-size: 0.9rem;
            padding: 0.5rem;
            border-top: 1px solid #eaeaea;
        }
        </style>
    
        <div class="footer">
            Badge styles credited to <a href="https://uiverse.io/" target="_blank">Uiverse.io</a>, 
            <a href="https://freefrontend.com/css-badges/" target="_blank">FreeFrontend</a>, and 
            <a href="https://codepen.io/" target="_blank">CodePen</a> creators: 
            <a href="https://codepen.io/jonnitto/pen/xQYEGV" target="_blank">jonnitto</a>, 
            <a href="https://codepen.io/zachacole/pen/xbzaJP" target="_blank">zachacole1</a>, 
            <a href="https://codepen.io/zachacole/pen/zxRxWM" target="_blank">zachacole2</a>,  
            <a href="https://codepen.io/gatauade/pen/zpVLzd" target="_blank">gatauade</a>.
        </div>
        """,
        unsafe_allow_html=True
    )
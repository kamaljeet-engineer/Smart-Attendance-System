import streamlit as st
from main import auto_capture_faces, train_model, mark_attendance
import os
from datetime import datetime
import pandas as pd

# ---------- CONFIG ----------
st.set_page_config(page_title="Smart Attendance", layout="wide")

# ---------- SESSION LOGIN ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------- LOGIN PAGE ----------
if not st.session_state.logged_in:

    st.title("🔐 Login - Smart Attendance")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "20272023":
            st.session_state.logged_in = True
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()

# ---------- WHITE UI ----------
st.markdown("""
<style>
.stApp { background-color: #f1f5f9; }
.block-container {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}
h1, h2, h3, h4, p { color: black !important; }
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("📸 Smart Attendance System")
st.markdown("### Face Recognition Based Attendance")

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Menu")

menu = st.sidebar.radio("Choose Action", [
    "🏠 Dashboard",
    "📷 Capture Face",
    "🧠 Train Model",
    "✅ Mark Attendance"
])

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------- DASHBOARD ----------
if menu == "🏠 Dashboard":

    st.success("Welcome to Smart Attendance System 🚀")

    col1, col2, col3 = st.columns(3)

    total_students = len(os.listdir("dataset"))
    col1.metric("👥 Total Students", total_students)

    today = datetime.now().strftime("%Y-%m-%d")
    file_path = f"attendance/attendance_{today}.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        total_today = len(df)
    else:
        df = pd.DataFrame(columns=["Name", "Time"])
        total_today = 0

    col2.metric("📅 Today's Attendance", total_today)
    col3.metric("🟢 System Status", "Active")

    st.markdown("---")

    # 🔍 SEARCH
    search = st.text_input("🔍 Search Student")

    if search:
        df = df[df["Name"].str.contains(search, case=False)]

    # 📋 TABLE
    st.subheader("📋 Today's Attendance")
    st.dataframe(df, use_container_width=True)

    # 📥 DOWNLOAD
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button("📥 Download Attendance", f, file_name=f"attendance_{today}.csv")

    # 📊 GRAPH
    st.subheader("📊 Attendance Graph")

    if not df.empty:
        chart_data = df["Name"].value_counts()
        st.bar_chart(chart_data)
    else:
        st.info("No data for chart")

# ---------- CAPTURE ----------
elif menu == "📷 Capture Face":

    st.header("📷 Capture New Face")

    name = st.text_input("Enter Student Name")

    if st.button("Start Capture"):
        if name.strip() == "":
            st.error("⚠️ Enter name first")
        else:
            st.warning("Opening camera...")

            import builtins
            original_input = builtins.input
            builtins.input = lambda _: name

            auto_capture_faces()

            builtins.input = original_input

            st.success("✅ Face Captured")

# ---------- TRAIN ----------
elif menu == "🧠 Train Model":

    st.header("🧠 Train Model")

    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            train_model()
        st.success("✅ Training Completed")

# ---------- ATTENDANCE ----------
elif menu == "✅ Mark Attendance":

    st.header("✅ Mark Attendance")

    if st.button("Start Attendance"):
        st.warning("Opening camera...")

        mark_attendance()

        st.success("✅ Attendance Completed")

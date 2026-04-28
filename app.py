import streamlit as st
from main import auto_capture_faces, train_model, mark_attendance

st.set_page_config(page_title="Smart Attendance", layout="centered")

st.title("📸 Smart Attendance System")

st.write("Select an option below:")

# Buttons UI
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📷 Capture Face"):
        st.warning("Camera will open...")
        auto_capture_faces()
        st.success("Face Capture Done ✅")

with col2:
    if st.button("🧠 Train Model"):
        st.warning("Training started...")
        train_model()
        st.success("Training Completed ✅")

with col3:
    if st.button("✅ Mark Attendance"):
        st.warning("Camera will open for attendance...")
        mark_attendance()
        st.success("Attendance Process Done ✅")
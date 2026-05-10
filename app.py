import streamlit as st
import cv2
import numpy as np
import faiss
import pickle
import os
from datetime import datetime, date, timedelta
import time
import sqlite3
import pandas as pd
import plotly.express as px

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="FaceVault", 
    layout="centered",
    page_icon="🔐",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1100px; }
    h1 { font-size: 2.3rem !important; margin-bottom: 0.3rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("FACE RECOGNITION BASED ATTENDANCE SYSTEM FROM CCTV LIVE")

# ========================== SESSION STATE ==========================
for key in ["face_system", "mode", "collected", "current_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "collected" and key != "mode" else [] if key == "collected" else "idle"

# ========================== IMPORT PIPELINE ==========================
from face_system import FaceAttendanceSystem

# Initialize System
if st.session_state.face_system is None:
    with st.spinner("Loading YOLO + MediaPipe + ArcFace Pipeline..."):
        st.session_state.face_system = FaceAttendanceSystem(threshold=0.55)
        # Clean inconsistent embeddings (Temporary fix)
        if st.session_state.face_system is not None:
            cleaned = [item for item in st.session_state.face_system.embeddings if len(item) >= 2]
            st.session_state.face_system.embeddings = cleaned

# ================== CLEAN BAD DATA ==================
if st.session_state.face_system is not None:
    cleaned = []
    for item in st.session_state.face_system.embeddings:
        if len(item) >= 3 and item[2] is not None:
            cleaned.append(item)
        
    st.session_state.face_system.embeddings = cleaned
    st.session_state.face_system._load_faiss()

# ========================== ATTENDANCE DB ==========================
def init_attendance_db():
    conn = sqlite3.connect("attendance.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    date TEXT,
                    user_id TEXT,
                    name TEXT)''')
    conn.commit()
    conn.close()

def mark_attendance(user_id, name):
    today = date.today().isoformat()
    conn = sqlite3.connect("attendance.db")
    existing = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=? AND user_id=?", 
                           (today, user_id)).fetchone()[0]
    if existing == 0:
        conn.execute("INSERT INTO attendance (timestamp, date, user_id, name) VALUES (?, ?, ?, ?)",
                    (datetime.now().isoformat(), today, user_id, name))
        conn.commit()
    conn.close()

def get_attendance_df():
    conn = sqlite3.connect("attendance.db")
    df = pd.read_sql("SELECT * FROM attendance", conn)
    conn.close()
    return df

def init_detection_log_db():
    conn = sqlite3.connect("attendance.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS detection_log (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    date TEXT,
                    name TEXT,
                    user_id TEXT,
                    confidence REAL)''')
    conn.commit()
    conn.close()


def log_detection(name, user_id=None, confidence=0.0):
    """Log detection with smart rules:
       - Registered person → Only ONCE per day
       - Unknown → Log every time (or at least multiple times)
    """
    today = date.today().isoformat()
    conn = sqlite3.connect("attendance.db")
    
    if name != "Unknown":
        # For registered users: Only once per day
        existing = conn.execute("""
            SELECT COUNT(*) FROM detection_log 
            WHERE date = ? AND name = ? AND name != 'Unknown'
        """, (today, name)).fetchone()[0]
        
        if existing > 0:
            conn.close()
            return False  # Already logged today
    else:
        # For Unknown: Allow multiple logs, but avoid flooding (once every 10 seconds)
        recent = conn.execute("""
            SELECT COUNT(*) FROM detection_log 
            WHERE name = 'Unknown' 
            AND timestamp > datetime('now', '-10 seconds')
        """).fetchone()[0]
        
        if recent > 0:
            conn.close()
            return False  # Already logged a Unknown very recently

    # Insert new log
    conn.execute("""INSERT INTO detection_log 
                    (timestamp, date, name, user_id, confidence)
                    VALUES (?, ?, ?, ?, ?)""",
                 (datetime.now().isoformat(), 
                  today, 
                  name, 
                  user_id, 
                  round(float(confidence), 4)))
    conn.commit()
    conn.close()
    return True


def get_detection_log_df():
    conn = sqlite3.connect("attendance.db")
    try:
        df = pd.read_sql("""
            SELECT timestamp, date, name, user_id, confidence 
            FROM detection_log 
            ORDER BY timestamp DESC
        """, conn)
    except:
        df = pd.DataFrame(columns=['timestamp', 'date', 'name', 'user_id', 'confidence'])
    finally:
        conn.close()
    return df

init_attendance_db()
init_detection_log_db()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "📹 Live Recognition", "📋 Register", 
    "👥 Manage", "📸 Image Register", "📋 Attendance Log"
])

# ===================== DASHBOARD =====================
with tab1:
    # ... (unchanged) ...
    st.subheader("Dashboard")
    df = get_attendance_df()
    today_str = date.today().isoformat()
    total_students = len([item for item in st.session_state.face_system.embeddings if len(item) >= 2])

    if not df.empty and 'date' in df.columns:
        today_present = len(df[df['date'] == today_str])
        last_7 = [(date.today() - timedelta(days=i)).isoformat() for i in range(7)]
        last_7_count = len(df[df['date'].isin(last_7)])
        avg_present = round(last_7_count / 7, 1)
    else:
        today_present = 0
        avg_present = 0.0

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("👥 Total Students", total_students)
    with col2: st.metric("✅ Today's Present", today_present)
    with col3: st.metric("❌ Today's Absent", total_students - today_present)

    col4, col5 = st.columns(2)
    with col4: st.metric("📅 Avg. Present (7 Days)", avg_present)
    with col5: st.metric("📉 Avg. Absent (7 Days)", round(total_students - avg_present, 1))

    st.subheader("Attendance Trend (Last 7 Days)")
    chart_type = st.radio("Chart Type", ["Bar", "Line"], horizontal=True)

    if not df.empty and 'date' in df.columns:
        daily = df.groupby('date').size().reset_index(name='count')
        daily['date'] = pd.to_datetime(daily['date'])
    else:
        daily = pd.DataFrame({'date': pd.date_range(end=date.today(), periods=7), 'count': [0]*7})

    full_range = pd.date_range(start=date.today() - timedelta(days=6), end=date.today())
    daily = daily.set_index('date').reindex(full_range, fill_value=0).reset_index()
    daily.columns = ['date', 'count']

    if chart_type == "Bar":
        fig = px.bar(daily, x='date', y='count', color_discrete_sequence=['#6366F1'])
    else:
        fig = px.line(daily, x='date', y='count', markers=True, line_shape='spline', color_discrete_sequence=['#22D3EE'])

    fig.update_layout(height=420, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ===================== LIVE RECOGNITION =====================
with tab2:
    st.subheader("Real-time Face Recognition")
    enable_cam = st.checkbox("Start Camera", value=True)
    frame_placeholder = st.empty()
    log_placeholder = st.empty()   # For live log feedback

    if enable_cam:
        cap = cv2.VideoCapture('Test.mp4')  # Change to 0 for webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error("Cannot access camera.")
        else:
            while enable_cam:
                ret, frame = cap.read()
                if not ret: 
                    break

                results = st.session_state.face_system.recognize(frame)
                vis = frame.copy()

                for res in results:
                    x1, y1, x2, y2 = res['bbox']
                    color = (0, 255, 0) if res['name'] != 'Unknown' else (0, 0, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, f"{res['name']} ({res['confidence']:.2f})", 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                    # Smart Logging
                    user_id = None
                    if res['name'] != 'Unknown':
                        for item in st.session_state.face_system.embeddings:
                            if len(item) >= 2 and item[1] == res['name']:
                                user_id = item[0]
                                mark_attendance(user_id, res['name'])
                                break
                    
                    # Log the detection
                    log_detection(res['name'], user_id, res.get('confidence', 0.0))
                    
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(vis_rgb, channels="RGB", use_container_width=True)
                time.sleep(0.03)
        
        cap.release()

# ===================== REGISTER =====================
with tab3:
    # ... (your existing registration code - unchanged) ...
    st.subheader("Register New Person")
    col1, col2 = st.columns([3, 1])
    with col1:
        name = st.text_input("👤 Full Name", placeholder="Enter full name")
    with col2:
        target = st.slider("Samples to collect", 10, 60, 25)

    if st.button("🚀 Start Registration", type="primary", use_container_width=True):
        if name.strip():
            st.session_state.current_name = name.strip()
            st.session_state.collected = []
            st.session_state.mode = "register"
            st.rerun()

    if st.session_state.mode == "register":
        st.info(f"**Registering:** {st.session_state.current_name}")
        cap = cv2.VideoCapture(0)
        frame_ph = st.empty()
        prog = st.progress(0)
        counter = st.empty()
        count = 0
        target_count = target

        while count < target_count:
            ret, frame = cap.read()
            if not ret: break

            faces_info = st.session_state.face_system.pipeline.detect_and_align(frame)
            vis = frame.copy()

            for face in faces_info:
                if count < target_count:
                    st.session_state.collected.append(face['embedding'])
                    count += 1
                    prog.progress(count / target_count)
                    counter.write(f"**Collected: {count} / {target_count}**")

                bbox = face['bbox']
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 100), 3)

            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            frame_ph.image(vis_rgb, channels="RGB", width="stretch")
            time.sleep(0.12)

        cap.release()

        if st.session_state.collected:
            embeddings = np.array(st.session_state.collected)
            num, pid = st.session_state.face_system.register_user_from_embeddings(
                embeddings, st.session_state.current_name)
            
            st.success(f"✅ Registration Successful!")
            st.balloons()
            st.info(f"**Name:** {st.session_state.current_name} | **ID:** {pid} | **Embeddings:** {num}")
            st.session_state.mode = "idle"
            st.session_state.collected = []
            st.session_state.current_name = ""
            time.sleep(2)
            st.rerun()

# ===================== MANAGE =====================
with tab4:
    st.subheader("Registered People")
    identities = st.session_state.face_system.embeddings
    
    if identities:
        # Safe data preparation
        data = []
        for item in identities:
            if len(item) >= 2:
                uid = item[0]
                name = item[1]
                data.append({"Name": name, "Person ID": uid})
        
        st.dataframe(data, use_container_width=True)

        # ==================== DELETE SECTION ====================
        st.subheader("🗑 Delete Identity")
        
        col_name, col_btn = st.columns([3, 1])
        
        with col_name:
            name_options = [item[1] for item in identities if len(item) >= 2]
            name_to_delete = st.selectbox(
                "Select person to delete", 
                name_options,
                key="delete_selectbox"
            )
        
        with col_btn:
            delete_clicked = st.button("🗑 Delete", type="secondary")

        confirm_delete = st.checkbox(
            f"⚠️ Confirm permanent deletion of **{name_to_delete}**", 
            key="confirm_delete_key"
        )

        if delete_clicked:
            if confirm_delete:
                with st.spinner(f"Deleting {name_to_delete}..."):
                    success = st.session_state.face_system.delete_identity(name_to_delete)
                    if success:
                        st.success(f"✅ **{name_to_delete}** deleted successfully")
                        if "delete_selectbox" in st.session_state:
                            del st.session_state.delete_selectbox
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete")
            else:
                st.warning("Please check the confirmation box")

        st.caption("⚠️ This action is permanent.")
    else:
        st.info("No registered faces yet.")

# ===================== ATTENDANCE LOG =====================
with tab6:
    st.subheader("📋 Full Detection Log (One Entry Per Person Per Day)")
    
    df = get_detection_log_df()
    
    st.divider()
    st.subheader("🗑️ Clear Logs")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Today's Logs**")
        if st.button("🗑️ Clear Today's Attendance", key="clear_att_today", use_container_width=True):
            today = date.today().isoformat()
            conn = sqlite3.connect("attendance.db")
            deleted = conn.execute("DELETE FROM attendance WHERE date=?", (today,)).rowcount
            conn.commit()
            conn.close()
            st.success(f"✅ Cleared {deleted} attendance records for today!")
            st.rerun()

    with col2:
        st.write("**Today's Detections**")
        if st.button("🗑️ Clear Today's Detection Log", key="clear_det_today", use_container_width=True):
            today = date.today().isoformat()
            conn = sqlite3.connect("attendance.db")
            deleted = conn.execute("DELETE FROM detection_log WHERE date=?", (today,)).rowcount
            conn.commit()
            conn.close()
            st.success(f"✅ Cleared {deleted} detection records (incl. Unknown) for today!")
            st.rerun()

    with col3:
        st.write("**All Logs**")
        if st.button("🗑️ Clear ALL Logs", key="clear_all", type="secondary", use_container_width=True):
            if st.checkbox("⚠️ Confirm: Delete **ALL** historical logs permanently?", key="confirm_all_clear"):
                conn = sqlite3.connect("attendance.db")
                att_deleted = conn.execute("DELETE FROM attendance").rowcount
                det_deleted = conn.execute("DELETE FROM detection_log").rowcount
                conn.commit()
                conn.close()
                st.success(f"🗑️ All logs cleared! ({att_deleted} attendance + {det_deleted} detections)")
                st.rerun()
            else:
                st.warning("Please check the confirmation box above")

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        df['Time'] = df['timestamp'].dt.strftime('%H:%M:%S')
        df['Date'] = df['timestamp'].dt.strftime('%d %b %Y')
        
        # Summary
        total = len(df)
        known = len(df[df['name'] != 'Unknown'])
        st.metric("Total Unique Detections Today", total, delta=f"{known} Known • {total-known} Unknown")
        
        # Group by Date
        st.divider()
        dates = sorted(df['date'].unique(), reverse=True)
        
        for d in dates[:10]:  # Show last 10 days
            day_df = df[df['date'] == d]
            with st.expander(f"📅 {d} — {len(day_df)} detections", expanded=(d == dates[0])):
                st.dataframe(
                    day_df[['Time', 'name', 'user_id', 'confidence']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "name": st.column_config.TextColumn("👤 Name", width="medium"),
                        "user_id": "ID",
                        "Time": "🕒 Time",
                        "confidence": st.column_config.NumberColumn("Confidence", format="%.3f")
                    }
                )
    else:
        st.info("No detections logged yet. Start the camera in Live Recognition tab.")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import base64

st.set_page_config(page_title="Classroom: Face & Engagement Prototype", layout="wide")

# Initialize session state
if "stats" not in st.session_state:
    st.session_state.stats = {
        "frames": 0,
        "engaged": 0,
        "not_engaged": 0,
        "last_engagement": "Unknown",
        "log": [],  # list of dicts per frame
    }

# Load Haar cascades
@st.cache_resource(show_spinner=False)
def load_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return face_cascade, eye_cascade

face_cascade, eye_cascade = load_cascades()

st.title("Student Identification & Engagement - Quick Prototype")
st.caption("Webcam snapshot-based demo using face detection and a simple engagement heuristic.")

with st.sidebar:
    st.header("Controls")
    draw_boxes = st.checkbox("Draw face boxes", value=True)
    show_eyes = st.checkbox("Try eye detection (slower)", value=False)
    min_face_percent = st.slider("Min face width (% of image width) to count as present", 5, 80, 10)
    center_tolerance = st.slider("Center tolerance (% of image width)", 5, 50, 15)

    if st.button("Reset session stats"):
        st.session_state.stats = {"frames": 0, "engaged": 0, "not_engaged": 0, "last_engagement": "Unknown", "log": []}
        st.success("Session stats reset.")

col_input, col_output = st.columns([1, 2])

with col_input:
    img = st.camera_input("Capture a frame (allow camera access)")
    st.info("Tip: capture multiple frames to build up engagement stats.")

with col_output:
    frame_placeholder = st.empty()
    info_placeholder = st.empty()


def compute_engagement(img_bgr, faces):
    h, w = img_bgr.shape[:2]
    engagement_label = "Not Engaged"
    score = 0.0

    if len(faces) == 0:
        return engagement_label, score

    # Use the largest face as the primary subject
    areas = [(x, y, fw, fh, fw * fh) for (x, y, fw, fh) in faces]
    x, y, fw, fh, _ = max(areas, key=lambda t: t[4])

    # Face size feature
    face_width_pct = (fw / w) * 100
    size_ok = face_width_pct >= min_face_percent

    # Centering feature
    face_cx = x + fw / 2
    img_cx = w / 2
    center_offset_pct = abs(face_cx - img_cx) / w * 100
    centered = center_offset_pct <= center_tolerance

    # Eye detection feature (optional)
    eyes_ok = True
    if show_eyes:
        roi_gray = cv2.cvtColor(img_bgr[y:y+fh, x:x+fw], cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        eyes_ok = len(eyes) >= 1

    # Simple scoring
    score = 0
    score += 0.5 if size_ok else 0
    score += 0.4 if centered else 0
    score += 0.1 if eyes_ok else 0

    engagement_label = "Engaged" if score >= 0.6 else "Not Engaged"
    return engagement_label, score


def draw_annotations(img_bgr, faces, engagement_label):
    out = img_bgr.copy()
    color = (0, 200, 0) if engagement_label == "Engaged" else (0, 0, 255)
    for (x, y, w, h) in faces:
        if draw_boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    cv2.putText(out, f"{engagement_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return out


if img is not None:
    # Convert uploaded image to OpenCV BGR
    pil_image = Image.open(img)
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    engagement_label, engagement_score = compute_engagement(bgr, faces)

    # Update stats
    st.session_state.stats["frames"] += 1
    if engagement_label == "Engaged":
        st.session_state.stats["engaged"] += 1
    else:
        st.session_state.stats["not_engaged"] += 1
    st.session_state.stats["last_engagement"] = engagement_label
    st.session_state.stats["log"].append({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "faces": int(len(faces)),
        "engagement": engagement_label,
        "score": round(float(engagement_score), 3),
    })

    annotated = draw_annotations(bgr, faces, engagement_label)
    frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Show stats
    s = st.session_state.stats
    engaged_pct = (s["engaged"] / s["frames"]) * 100 if s["frames"] else 0
    info_placeholder.markdown(
        f"""
        - Frames analyzed: {s['frames']}
        - Faces detected (latest): {len(faces)}
        - Last engagement: {s['last_engagement']} (score: {engagement_score:.2f})
        - Engagement rate: {engaged_pct:.1f}%
        """
    )

    # Download log as CSV
    if s["log"]:
        import csv
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["ts", "faces", "engagement", "score"])
        writer.writeheader()
        writer.writerows(s["log"])
        st.download_button("Download session log (CSV)", data=output.getvalue(), file_name="engagement_log.csv", mime="text/csv")
else:
    frame_placeholder.info("Waiting for a camera snapshot...")
    info_placeholder.empty()

st.markdown("---")
st.subheader("Notes")
st.markdown(
    "- This is a prototype: it detects faces and infers a simple engagement score based on face size, centering, and optional eye detection.\n"
    "- For identification and robust engagement modeling, we'll integrate a face recognition model (e.g., DeepFace) and fine-tune an engagement model on classroom data next."
)
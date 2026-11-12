"""
Dashboard Klasifikasi Emosi Wajah (CNN) — Streamlit
Jalankan:  streamlit run app.py
Pastikan notebook emotion_recognition.ipynb sudah dijalankan sampai selesai (folder output/ terisi).
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------
st.set_page_config(page_title="Emotion Recognition CNN", page_icon="🙂", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
MODEL_PATH = os.path.join(OUT, "final_emotion_cnn_model.keras")

EMOJI = {"angry": "😠", "anger": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
         "happiness": "😄", "sad": "😢", "sadness": "😢", "surprise": "😲",
         "surprised": "😲", "neutral": "😐", "contempt": "😒"}
CONF_THRESHOLD = 50  # % — di bawah ini tampilkan peringatan

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white;
           padding:1.6rem 1.5rem; border-radius:16px; margin-bottom:1.2rem;}
    .hero h1 {margin:0; font-size:2rem;}
    .hero p {margin:.4rem 0 0; opacity:.92;}
    .result-card {background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%); color:#06352c;
           padding:1.2rem; border-radius:16px; text-align:center;}
    .result-card .emoji {font-size:3.2rem;}
    .result-card .name {font-size:1.8rem; font-weight:700; text-transform:capitalize;}
    .result-card .conf {font-size:1.05rem;}
    </style>
    <div class="hero">
      <h1>🙂 Klasifikasi Emosi Wajah dengan CNN</h1>
      <p>Praktikum Analisis Big Data — Convolutional Neural Network (Keras) + Streamlit.
         Dashboard ini menampilkan hasil eksperimen dan memprediksi emosi dari foto wajah baru.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

REQUIRED = ["class_names.json", "config.json", "history.json", "dataset_info.json",
            "metrics.json", "final_emotion_cnn_model.keras"]
missing = [f for f in REQUIRED if not os.path.exists(os.path.join(OUT, f))]
if missing:
    st.error("File berikut belum ada di folder `output/`: " + ", ".join(missing) +
             ". Jalankan dulu `emotion_recognition.ipynb` sampai selesai.")
    st.stop()


def read_json(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()
class_names = read_json("class_names.json")
cfg = read_json("config.json")
history = read_json("history.json")
info = read_json("dataset_info.json")
metrics = read_json("metrics.json")
IMG_SIZE = tuple(cfg["img_size"])


def emo(c):
    return EMOJI.get(c.lower(), "🙂")


def preprocess(img: Image.Image):
    """Grayscale + resize. Skala tetap 0-255 (normalisasi 0-1 sudah ada di dalam model)."""
    img = img.convert("L").resize(IMG_SIZE)
    return np.asarray(img, dtype="float32")[None, :, :, None]


def grad_cam(arr, class_idx=None):
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(cfg["last_conv_layer"]).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model([arr], training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
    return cam / (cam.max() + 1e-8)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.write(f"**Arsitektur:** {cfg['architecture']}")
    st.write(f"**Input:** {IMG_SIZE[0]}×{IMG_SIZE[1]} piksel ({cfg['color_mode']})")
    st.write(f"**Optimizer:** {cfg['optimizer']} (lr={cfg['learning_rate']})")
    st.write(f"**Batch size:** {cfg['batch_size']}")
    st.write(f"**Epoch dijalankan:** {cfg['epochs_run']} (terbaik: {cfg['best_epoch']})")
    st.markdown("**Kelas emosi:**")
    for c in class_names:
        st.write(f"{emo(c)} {c.capitalize()}")
    st.divider()
    st.caption("Tips: gunakan foto wajah tampak depan dengan pencahayaan cukup.")

tab_data, tab_train, tab_eval, tab_pred = st.tabs(
    ["📊 Dataset", "📈 Hasil Pelatihan", "🧪 Evaluasi Model", "🔮 Prediksi Gambar Baru"])

# ------------------------------------------------------------
# TAB 1 — DATASET
# ------------------------------------------------------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah kelas", len(class_names))
    c2.metric("Total gambar", f"{info['total']:,}")
    c3.metric("Data latih", f"{info['n_train']:,}")
    c4.metric("Data uji", f"{info['n_test']:,}")
    st.caption(f"Proporsi latih : uji = {info['n_train'] / info['total'] * 100:.0f}% : "
               f"{info['n_test'] / info['total'] * 100:.0f}%")

    st.subheader("Distribusi gambar per kelas")
    dist_df = pd.DataFrame({"Latih": info["train_counts"], "Uji": info["test_counts"]})
    st.bar_chart(dist_df)
    ratio = max(info["class_counts"].values()) / max(min(info["class_counts"].values()), 1)
    if ratio > 1.5:
        st.warning(f"Terdapat class imbalance (rasio kelas terbanyak : tersedikit = {ratio:.1f}×). "
                   "Ditangani dengan class weight saat training.")
    else:
        st.success("Distribusi kelas relatif seimbang.")
    st.dataframe(pd.Series(info["class_counts"], name="Jumlah gambar").to_frame())

    p = os.path.join(OUT, "1b_sample_per_class.png")
    if os.path.exists(p):
        st.subheader("Contoh gambar tiap emosi")
        st.image(p)

# ------------------------------------------------------------
# TAB 2 — TRAINING
# ------------------------------------------------------------
with tab_train:
    ep = list(range(1, len(history["accuracy"]) + 1))
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Akurasi")
        st.line_chart(pd.DataFrame({"Train": history["accuracy"], "Validasi": history["val_accuracy"]}, index=ep))
    with colB:
        st.subheader("Loss")
        st.line_chart(pd.DataFrame({"Train": history["loss"], "Validasi": history["val_loss"]}, index=ep))
    st.caption(f"Epoch terbaik: {cfg['best_epoch']} · Val accuracy tertinggi: {max(history['val_accuracy']) * 100:.2f}%")

# ------------------------------------------------------------
# TAB 3 — EVALUASI
# ------------------------------------------------------------
with tab_eval:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    m2.metric("Precision (macro)", f"{metrics['macro']['precision']:.3f}")
    m3.metric("Recall (macro)", f"{metrics['macro']['recall']:.3f}")
    m4.metric("F1-score (macro)", f"{metrics['macro']['f1']:.3f}")

    st.subheader("Metrik per kelas")
    pc = pd.DataFrame(metrics["per_class"]).T.rename(
        columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-score", "support": "Jumlah uji"})
    pc["Jumlah uji"] = pc["Jumlah uji"].astype(int)
    st.dataframe(pc.round(3))
    best, worst = pc["F1-score"].idxmax(), pc["F1-score"].idxmin()
    st.info(f"Paling mudah dikenali: **{best}** (F1 {pc.loc[best, 'F1-score']:.3f}) · "
            f"Paling sulit: **{worst}** (F1 {pc.loc[worst, 'F1-score']:.3f})")

    st.subheader("Confusion matrix")
    cm = np.array(metrics["confusion_matrix"])
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Proporsi")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm_pct[i, j] > .5 else "black")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Kelas sebenarnya")
    plt.tight_layout()
    st.pyplot(fig)

    for fname, title in [("4_sample_predictions.png", "Contoh prediksi pada data uji"),
                         ("5_gradcam.png", "Grad-CAM — area wajah yang paling memengaruhi keputusan model")]:
        pth = os.path.join(OUT, fname)
        if os.path.exists(pth):
            st.subheader(title)
            st.image(pth)

# ------------------------------------------------------------
# TAB 4 — PREDIKSI
# ------------------------------------------------------------
with tab_pred:
    mode = st.radio("Sumber gambar", ["📁 Upload", "📷 Kamera"], horizontal=True)
    if mode == "📁 Upload":
        file = st.file_uploader("Pilih foto wajah", type=["jpg", "jpeg", "png", "webp"])
    else:
        file = st.camera_input("Ambil foto wajah")

    if file is None:
        st.info("Belum ada gambar. Upload foto atau gunakan kamera untuk memulai.")
    else:
        image = Image.open(file)
        arr = preprocess(image)
        probs = model.predict(arr, verbose=0)[0]
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        name, conf = class_names[top], float(probs[top]) * 100

        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.image(image, caption="Gambar input")
        with col2:
            st.markdown(
                f"""<div class="result-card"><div class="emoji">{emo(name)}</div>
                <div class="name">{name}</div><div class="conf">Keyakinan: {conf:.1f}%</div></div>""",
                unsafe_allow_html=True)
            if conf < CONF_THRESHOLD:
                st.warning("Keyakinan rendah. Coba foto yang lebih jelas / wajah tampak depan.")
            else:
                st.success("Prediksi cukup meyakinkan.")
        with col3:
            cam = grad_cam(arr, top)
            cam_big = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, Image.BILINEAR)) / 255.
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            ax2.imshow(arr[0, :, :, 0], cmap="gray", vmin=0, vmax=255)
            ax2.imshow(cam_big, cmap="jet", alpha=.45)
            ax2.axis("off")
            ax2.set_title("Grad-CAM", fontsize=10)
            st.pyplot(fig2)

        st.subheader("Probabilitas semua kelas")
        st.bar_chart(pd.Series({f"{emo(class_names[i])} {class_names[i]}": float(probs[i]) * 100
                                for i in order}, name="Probabilitas (%)"))

st.divider()
st.caption("Dibuat dengan TensorFlow/Keras & Streamlit")

# app/app.py
import pandas as pd
import os
import streamlit as st
import torch, numpy as np, cv2, os, sys, tempfile
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model        import build_model
from src.dataset      import get_transforms
from src.gradcam      import (generate_gradcam, analyze_heatmap_regions,
                               multi_method_comparison)
from src.lime_explain import (generate_lime_explanation, lime_heatmap,
                               plot_lime_analysis)
from src.report       import (save_combined_figure, generate_pdf_report,
                               DR_CLASSES, RISK_LEVELS, DR_DESCRIPTIONS)

st.set_page_config(page_title="DR Screener", page_icon="👁", layout="wide")

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'models/best_model.pth'
STAGE_COLORS = {0:'#27ae60', 1:'#f1c40f', 2:'#e67e22',
                3:'#e74c3c', 4:'#8e44ad'}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    patient_id   = st.text_input("Patient ID", "DR-001")
    cam_method   = st.selectbox("Grad-CAM method",
                                 ["gradcam++","gradcam","eigencam"])
    run_lime     = st.checkbox("Run LIME", value=True)
    lime_samples = st.slider("LIME samples", 200, 1500, 600, step=100)
    show_compare = st.checkbox("Show multi-method XAI comparison", value=False)
    st.divider()
    st.caption(f"Running on: **{DEVICE.upper()}**")
    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found at models/best_model.pth — "
                   "train the model first.")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = build_model(num_classes=5, device=DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Diabetic Retinopathy AI Screener")
st.caption("Upload a retinal fundus image to get an AI prediction "
           "with Grad-CAM++ and LIME explanations.")

uploaded = st.file_uploader("Choose a fundus image",
                              type=["png","jpg","jpeg"])

if uploaded and os.path.exists(MODEL_PATH):
    pil_img = Image.open(uploaded).convert('RGB')
    img_np  = np.array(pil_img)

    # Preprocess
    transform = get_transforms('test', 224)
    tensor    = transform(image=img_np)['image']

    model = load_model()

    # Predict
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(DEVICE))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = int(probs.argmax())

    # ── Layout: image | result ────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.image(pil_img, caption="Uploaded image", use_column_width=True)

    with col2:
        color = STAGE_COLORS[pred_class]
        st.markdown(
            f"<div style='background:{color};padding:14px 18px;"
            f"border-radius:10px;color:white;margin-bottom:12px'>"
            f"<div style='font-size:20px;font-weight:600'>"
            f"{DR_CLASSES[pred_class]}</div>"
            f"<div style='font-size:13px;opacity:0.9'>"
            f"Confidence {probs[pred_class]*100:.1f}% &nbsp;|&nbsp; "
            f"Risk: {RISK_LEVELS[pred_class]}</div>"
            f"</div>", unsafe_allow_html=True)

        # Probability bars
        for i, (cls, p) in enumerate(zip(DR_CLASSES, probs)):
            c = color if i == pred_class else '#bdc3c7'
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;"
                f"margin:3px 0;font-size:13px'>"
                f"<span style='width:120px'>{cls}</span>"
                f"<div style='flex:1;background:#ecf0f1;border-radius:4px;height:16px'>"
                f"<div style='width:{p*100:.1f}%;background:{c};"
                f"height:100%;border-radius:4px'></div></div>"
                f"<span style='width:42px;text-align:right'>{p*100:.1f}%</span>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.info(DR_DESCRIPTIONS[pred_class])

    # ── XAI Section ───────────────────────────────────────────────────────────
    st.markdown("### Explainability Analysis")

    with st.spinner(f"Generating {cam_method.upper()} heatmap…"):
        img_resized  = cv2.resize(img_np, (224, 224))
        overlay, raw_cam, _, _ = generate_gradcam(
            model, tensor, pred_class,
            method=cam_method, device=DEVICE)
        active_regions = analyze_heatmap_regions(raw_cam)

    lime_overlay = None
    lime_exp     = None
    if run_lime:
        with st.spinner("Running LIME (this takes ~20–40 seconds)…"):
            lime_exp, lime_overlay, top_segs = generate_lime_explanation(
                model, img_resized, pred_class,
                device=DEVICE, num_samples=lime_samples)

    c1, c2, c3 = st.columns(3)
    c1.image(img_resized,  caption="Original",          use_column_width=True)
    c2.image(overlay,      caption=f"{cam_method.upper()} heatmap",
             use_column_width=True)
    if lime_overlay is not None:
        c3.image(lime_overlay, caption="LIME regions",  use_column_width=True)
    else:
        c3.markdown("*(LIME disabled)*")

    if active_regions:
        st.markdown(
            f"**Model attention focused on:** {', '.join(active_regions)}")

    # Multi-method comparison
    if show_compare:
        with st.spinner("Generating XAI method comparison…"):
            fig = multi_method_comparison(
                model, tensor, pred_class, device=DEVICE)
            st.pyplot(fig)

    # ── PDF Report ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Download Clinical Report")
    if st.button("Generate PDF report"):
        with st.spinner("Building PDF…"):
            with tempfile.TemporaryDirectory() as tmp:
                fig_path = os.path.join(tmp, 'fig.png')
                save_combined_figure(
                    img_resized, overlay,
                    lime_overlay if lime_overlay is not None else overlay,
                    pred_class, probs, active_regions, fig_path)

                pdf_path = os.path.join(tmp, 'report.pdf')
                generate_pdf_report(
                    patient_id, uploaded.name, fig_path,
                    pred_class, probs, active_regions, pdf_path)

                with open(pdf_path, 'rb') as f:
                    st.download_button(
                        "Download PDF",
                        data=f.read(),
                        file_name=f"DR_Report_{patient_id}.pdf",
                        mime="application/pdf")

elif uploaded and not os.path.exists(MODEL_PATH):
    st.error("Please train the model first (run `python src/main_train.py`), "
             "then reload this app.")
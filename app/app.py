"""Fresh-versus-Rotten Fruit Classifier — Streamlit prototype (3D themed UI).

Upload a photo OR use your webcam; the app shows the predicted fruit + freshness,
a confidence ring, animated probability bars, and inference latency — all wrapped
in a custom glassmorphism / 3D theme injected via CSS.
"""
import streamlit as st
from PIL import Image

from model_utils import load_model, predict

st.set_page_config(page_title="Fresh vs Rotten Fruit Classifier",
                   page_icon="🍎", layout="centered")

# ----------------------------------------------------------------- theme (CSS)
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

/* ---- app background: animated gradient + floating 3D orbs ---- */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 10% -10%, #1e2a5a 0%, transparent 55%),
              radial-gradient(1000px 700px at 110% 10%, #3a1a5e 0%, transparent 50%),
              linear-gradient(160deg, #0b1020 0%, #0e1430 50%, #0b1020 100%);
  background-attachment: fixed;
}
[data-testid="stHeader"]{ background: transparent; }
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after{
  content:""; position:fixed; border-radius:50%; filter:blur(60px);
  z-index:0; pointer-events:none; opacity:.55;
}
[data-testid="stAppViewContainer"]::before{
  width:420px; height:420px; top:-80px; right:-60px;
  background:radial-gradient(circle at 30% 30%, #ff7eb3, #ff4d6d 60%, transparent 70%);
  animation: float1 14s ease-in-out infinite;
}
[data-testid="stAppViewContainer"]::after{
  width:360px; height:360px; bottom:-100px; left:-80px;
  background:radial-gradient(circle at 30% 30%, #6ee7b7, #22c55e 60%, transparent 70%);
  animation: float2 18s ease-in-out infinite;
}
@keyframes float1{ 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-40px,50px) scale(1.12)} }
@keyframes float2{ 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(50px,-40px) scale(1.08)} }

.block-container{ position:relative; z-index:1; max-width:780px; padding-top:2rem; }

/* ---- typography ---- */
html, body, [class*="css"]{ font-family:'Inter',sans-serif; color:#e7ecff; }
h1,h2,h3{ font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.02em; }

/* ---- hero ---- */
.hero{ text-align:center; padding:14px 0 6px; perspective:900px; }
.hero .emoji{
  font-size:70px; display:inline-block;
  filter:drop-shadow(0 18px 22px rgba(0,0,0,.45));
  animation: bob 4s ease-in-out infinite; transform-style:preserve-3d;
}
@keyframes bob{ 0%,100%{transform:translateY(0) rotateZ(-4deg)} 50%{transform:translateY(-14px) rotateZ(4deg)} }
.hero h1{
  margin:.1em 0 .1em; font-size:2.5rem; font-weight:700;
  background:linear-gradient(92deg,#ff6b6b,#ffd93d 40%,#6ee7b7 75%,#60a5fa);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow:0 2px 30px rgba(110,231,183,.25);
}
.hero p{ color:#9fb0d9; font-size:.95rem; margin:0 auto; max-width:560px; }

/* ---- glass cards ---- */
.glass{
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);
  border-radius:22px; padding:22px; margin:18px 0;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.12);
}

/* ---- result card with 3D tilt ---- */
.result{
  text-align:center; transform-style:preserve-3d;
  transition:transform .4s cubic-bezier(.2,.8,.2,1), box-shadow .4s;
  animation: rise .6s cubic-bezier(.2,.8,.2,1);
}
.result:hover{ transform:perspective(1100px) rotateX(5deg) rotateY(-6deg) translateY(-6px) scale(1.01); }
@keyframes rise{ from{opacity:0; transform:translateY(24px)} to{opacity:1; transform:translateY(0)} }
.result.fresh{ box-shadow:0 20px 50px rgba(34,197,94,.30), inset 0 1px 0 rgba(255,255,255,.14); border-color:rgba(110,231,183,.45); }
.result.rotten{ box-shadow:0 20px 50px rgba(239,68,68,.30), inset 0 1px 0 rgba(255,255,255,.14); border-color:rgba(248,113,113,.45); }
.result .bigemoji{ font-size:64px; filter:drop-shadow(0 14px 18px rgba(0,0,0,.45)); animation:bob 4s ease-in-out infinite; }
.result .label{ font-family:'Space Grotesk'; font-size:1.9rem; font-weight:700; margin:6px 0 2px; }
.result .sub{ color:#9fb0d9; font-size:.9rem; letter-spacing:.14em; text-transform:uppercase; }

/* ---- confidence ring (conic gradient) ---- */
.ring-wrap{ display:flex; justify-content:center; margin:16px 0 6px; }
.ring{
  width:140px; height:140px; border-radius:50%; display:grid; place-items:center;
  position:relative; animation:pop .6s cubic-bezier(.2,1.3,.4,1);
}
@keyframes pop{ from{transform:scale(.6); opacity:0} to{transform:scale(1); opacity:1} }
.ring::before{ content:""; position:absolute; inset:12px; border-radius:50%; background:#0e1430; }
.ring .val{ position:relative; font-family:'Space Grotesk'; font-size:1.7rem; font-weight:700; }
.ring .cap{ position:relative; font-size:.62rem; color:#9fb0d9; letter-spacing:.18em; margin-top:-4px; }

/* ---- probability bars ---- */
.bars{ margin-top:4px; }
.bar-row{ margin:9px 0; }
.bar-top{ display:flex; justify-content:space-between; font-size:.86rem; color:#cdd7f5; margin-bottom:4px; }
.bar-track{ height:12px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden; }
.bar-fill{
  height:100%; border-radius:99px;
  background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
  box-shadow:0 0 14px rgba(167,139,250,.5);
  animation:grow 1.1s cubic-bezier(.2,.8,.2,1) forwards; width:0;
}
.bar-fill.top{ background:linear-gradient(90deg,#34d399,#22c55e); box-shadow:0 0 16px rgba(34,197,94,.6); }
@keyframes grow{ to{ width:var(--w); } }

/* ---- skin streamlit widgets ---- */
[data-testid="stFileUploaderDropzone"]{
  background:rgba(255,255,255,.05); border:1.5px dashed rgba(255,255,255,.25); border-radius:16px;
}
.stTabs [data-baseweb="tab-list"]{ gap:8px; }
.stTabs [data-baseweb="tab"]{
  background:rgba(255,255,255,.05); border-radius:12px 12px 0 0; padding:8px 16px;
}
.stTabs [aria-selected="true"]{ background:rgba(110,231,183,.14); }
[data-testid="stImage"] img{ border-radius:16px; box-shadow:0 14px 34px rgba(0,0,0,.45); }
#MainMenu, footer{ visibility:hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

FRUIT_EMOJI = {"apple": "🍎", "banana": "🍌", "orange": "🍊"}


def emoji_for(label: str) -> str:
    low = label.lower()
    for k, e in FRUIT_EMOJI.items():
        if k in low:
            return e
    return "🍏"


# ----------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
      <div class="emoji">🍎</div>
      <h1>Fresh vs Rotten Fruit Classifier</h1>
      <p>MobileNetV3 · 6 classes (Apple / Banana / Orange × Fresh / Rotten)<br>
      Deep Learning Lab Capstone · Ramdeobaba University</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model…")
def _get_model():
    return load_model()


try:
    model, labels = _get_model()
    model_ready = True
except FileNotFoundError as e:
    model_ready = False
    st.error(str(e))

# ----------------------------------------------------------------- input
st.markdown('<div class="glass">', unsafe_allow_html=True)
tab_upload, tab_camera = st.tabs(["📁  Upload image", "📷  Use webcam"])
image = None
with tab_upload:
    up = st.file_uploader("Upload a fruit image", type=["png", "jpg", "jpeg"],
                          label_visibility="collapsed")
    if up is not None:
        image = Image.open(up)
with tab_camera:
    cam = st.camera_input("Take a photo", label_visibility="collapsed")
    if cam is not None:
        image = Image.open(cam)
st.markdown('</div>', unsafe_allow_html=True)


def render_result(pretty, conf, prob_map, latency_ms):
    fresh = pretty.lower().startswith("fresh")
    accent = "#22c55e" if fresh else "#ef4444"
    deg = int(round(conf * 360))
    ring = (f"background:conic-gradient({accent} {deg}deg, rgba(255,255,255,.10) 0deg);")
    cls = "fresh" if fresh else "rotten"
    status = "✅ FRESH" if fresh else "⚠️ ROTTEN"

    st.markdown(
        f"""
        <div class="glass result {cls}">
          <div class="bigemoji">{emoji_for(pretty)}</div>
          <div class="label">{pretty}</div>
          <div class="sub">{status}</div>
          <div class="ring-wrap">
            <div class="ring" style="{ring}">
              <div class="val" style="color:{accent}">{conf*100:.0f}%</div>
              <div class="cap">CONFIDENCE</div>
            </div>
          </div>
          <div style="color:#9fb0d9;font-size:.82rem">⚡ {latency_ms:.0f} ms ·
             ~{1000/latency_ms:.0f} FPS inference (CPU)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # probability bars
    rows = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    bars = '<div class="glass bars">'
    for i, (name, p) in enumerate(rows):
        top = "top" if i == 0 else ""
        bars += (
            f'<div class="bar-row"><div class="bar-top"><span>{name}</span>'
            f'<span>{p*100:.1f}%</span></div>'
            f'<div class="bar-track"><div class="bar-fill {top}" '
            f'style="--w:{p*100:.1f}%"></div></div></div>'
        )
    bars += "</div>"
    st.markdown(bars, unsafe_allow_html=True)


if image is not None and model_ready:
    st.image(image, caption="Input image", use_container_width=True)
    with st.spinner("Classifying…"):
        pretty, conf, prob_map, latency_ms = predict(model, labels, image)
    render_result(pretty, conf, prob_map, latency_ms)
elif image is not None and not model_ready:
    st.image(image, caption="Input image (model not loaded)", use_container_width=True)

with st.expander("ℹ️  About this prototype"):
    st.markdown(
        "- **Model:** MobileNetV3 fine-tuned via transfer learning (96.4% test accuracy).\n"
        "- **Preprocessing:** resize → center-crop → ImageNet normalization.\n"
        "- **Classes:** Fresh/Rotten × Apple/Banana/Orange.\n"
        "- Trained on a balanced fruit dataset (70/15/15 split) and benchmarked "
        "against a baseline CNN."
    )

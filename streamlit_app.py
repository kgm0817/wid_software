# streaming_app.py
import streamlit as st
from PIL import Image
import numpy as np
from skimage import color, img_as_ubyte
import requests

# ───────── 공통 유틸 ────────────────────────────────────────────────────
def cvt_lab(rgb):          # np.uint8 RGB → skimage LAB(float, L 0‑100)
    return color.rgb2lab(rgb.astype(np.float32) / 255.0)

def cvat_rle_to_binary_image_mask(cvat_rle: dict, img_h: int, img_w: int) -> np.ndarray:
    # convert CVAT tight object RLE to COCO-style whole image mask
    rle = cvat_rle['rle']
    left = cvat_rle['left']
    top = cvat_rle['top']
    width = cvat_rle['width']

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    value = 0
    offset = 0
    for rle_count in rle:
        while rle_count > 0:
            y, x = divmod(offset, width)
            mask[y + top][x + left] = value
            rle_count -= 1
            offset += 1
        value = 1 - value

    return mask

@st.cache_data
def wid_from_lab(lab, mask):
    L,A,B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    return (0.511*L - 2.324*A - 1.100*B)[mask]

API_URL = "https://ai.linkdens-backend.com/api/segmentation/cam/white/tooth"


# ───────── Adjust 전용 UI & 로직 ───────────────────────────────────────
def run_adjust():
    uploaded = st.file_uploader("Upload an image", type=["jpg","png","bmp","tif"],
                                key="adj_upload")
    if not uploaded:
        st.info("Please upload an image.")
        return

    # 최초 업로드 또는 다른 파일이면 상태 리셋
    if st.session_state.get("adj_prev") != uploaded:
        st.session_state["adj_prev"]   = uploaded
        st.session_state["adj_rgb"]    = np.array(Image.open(uploaded).convert("RGB"))
        st.session_state["adj_mask"]   = None     # 아직 세그멘테이션 안 함
    rgb = st.session_state["adj_rgb"]

    if st.button("Run segmentation", key="adj_seg_btn"):
        with st.spinner("Calling API…"):
            resp = requests.post(API_URL, files={"file": uploaded.getvalue()})
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}")
            return
        anns = resp.json()
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        for ann in anns:
            info = {
                "rle": ann["rle"],
                "left": ann["boundingBox"]["point0"]["x"],
                "top":  ann["boundingBox"]["point0"]["y"],
                "width": ann["boundingBox"]["point1"]["x"] - ann["boundingBox"]["point0"]["x"] + 1,
            }
            mask |= cvat_rle_to_binary_image_mask(info, h, w).astype(bool)
        st.session_state["adj_mask"] = mask

    mask = st.session_state.get("adj_mask")
    if mask is None:
        st.warning("Click **Run segmentation** first.")
        return

    # ΔWI + 전략 → ΔLab
    col_wi, col_s = st.columns([1,1])
    dwi = col_wi.slider("ΔWI", -30.0, 30.0, 0.0, key="adj_dwi")
    strategy = col_s.selectbox("Strategy", ["L_only", "L2_min", "weighted"],
                               index=2, key="adj_strategy")

    coeff = np.array([0.511, -2.324, -1.100])
    if strategy == "L_only":
        dL, dA, dB = dwi/coeff[0], 0.0, 0.0
    elif strategy == "L2_min":
        scale      = dwi / np.dot(coeff, coeff)
        dL, dA, dB = scale * coeff
    else:                       # weighted
        winv       = np.array([1, 1/8, 1/8])
        scale      = dwi / np.dot(coeff*winv, coeff)
        dL, dA, dB = scale * coeff * winv
    dL = float(np.clip(dL, -50, 50))
    dA = float(np.clip(dA, -40, 40))
    dB = float(np.clip(dB, -40, 40))

    # Read‑only ΔLab 슬라이더
    colL,colA,colB = st.columns(3)
    colL.slider("ΔL", -50.0, 50.0, value=dL, disabled=True, key="adj_dL")
    colA.slider("Δa", -40.0, 40.0, value=dA, disabled=True, key="adj_dA")
    colB.slider("Δb", -40.0, 40.0, value=dB, disabled=True, key="adj_dB")

    # ROI 보정
    lab_orig = cvt_lab(rgb)
    lab_adj  = lab_orig.copy()
    for chan, delta, lo, hi in zip([0,1,2], [dL,dA,dB], [0,-128,-128], [100,127,127]):
        arr = lab_adj[:,:,chan]; arr[mask] = np.clip(arr[mask]+delta, lo, hi)
    rgb_adj = img_as_ubyte(color.lab2rgb(lab_adj))
    comp    = np.where(mask[:,:,None], rgb_adj, rgb)

    # 결과 표시
    c1,c2 = st.columns(2)
    c1.image(rgb, caption="Original", use_container_width=True)
    c2.image(comp, caption="Adjusted ROI", use_container_width=True)

    wi_orig = wid_from_lab(lab_orig, mask)
    wi_adj  = wid_from_lab(lab_adj , mask)
    c1.metric("WI median", f"{np.median(wi_orig):.2f}")
    c2.metric("WI median", f"{np.median(wi_adj ):.2f}")

# ───────── Compare 전용 UI & 로직 ──────────────────────────────────────
from streamlit_image_comparison import image_comparison

def run_compare():
    col_up1, col_up2 = st.columns(2)
    upl1 = col_up1.file_uploader("Before whitening", type=["jpg","png","bmp","tif"], key="cmp1_up")
    upl2 = col_up2.file_uploader("After  whitening", type=["jpg","png","bmp","tif"], key="cmp2_up")
    if not upl1 or not upl2:
        st.info("Upload both images.")
        return

    # ── 세그멘테이션 + WID -------------------------------------------------
    @st.cache_data(show_spinner=False)
    def process_uploaded(uploaded_bytes):
        rgb = np.array(Image.open(uploaded_bytes).convert("RGB"))
        h, w = rgb.shape[:2]
        resp = requests.post(API_URL, files={"file": uploaded_bytes.getvalue()})
        anns = resp.json()

        mask = np.zeros((h, w), dtype=bool)
        for ann in anns:
            info = {
                "rle":    ann["rle"],
                "left":   ann["boundingBox"]["point0"]["x"],
                "top":    ann["boundingBox"]["point0"]["y"],
                "width":  ann["boundingBox"]["point1"]["x"]
                        - ann["boundingBox"]["point0"]["x"] + 1,
            }
            mask |= cvat_rle_to_binary_image_mask(info, h, w).astype(bool)

        lab  = cvt_lab(rgb)
        wi   = wid_from_lab(lab, mask)

        masked = rgb.copy()
        masked[~mask] = 0                 # ROI만 남김

        return rgb, masked, wi            # ⬅︎ 3 개 반환

    # 이미지·WI 얻기
    orig1, mask1, wi1 = process_uploaded(upl1)
    orig2, mask2, wi2 = process_uploaded(upl2)

    # ---------- 공통 크기 맞추기 ----------
    h_max = max(orig1.shape[0], orig2.shape[0])
    w_max = max(orig1.shape[1], orig2.shape[1])

    def pad(img):
        ph = (h_max - img.shape[0]) // 2
        pw = (w_max - img.shape[1]) // 2
        canvas = np.zeros((h_max, w_max, 3), dtype=img.dtype)
        canvas[ph:ph+img.shape[0], pw:pw+img.shape[1]] = img
        return canvas

    disp1 = pad(orig1)
    disp2 = pad(orig2)
    mask1_p = pad(mask1)
    mask2_p = pad(mask2)

    disp_width = min(w_max, 450)

    # WI Median (왼쪽 · 오른쪽 정렬)
    lcol, rcol = st.columns([1, 1])

    # 공통 CSS 한 번만 주입
    st.markdown(
        """
        <style>
        .wi-box   {font-family: sans-serif; line-height:1.1;}
        .wi-label {font-size:16px; font-weight:bold;}
        .wi-val   {font-size:24px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    lcol.markdown(
        f"<div class='wi-box' style='text-align:left'>"
        f"<span class='wi-label'>Before WI median</span><br>"
        f"<span class='wi-val'>{np.median(wi1):.2f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    rcol.markdown(
        f"<div class='wi-box' style='text-align:right'>"
        f"<span class='wi-label'>After WI median</span><br>"
        f"<span class='wi-val'>{np.median(wi2):.2f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 3‑열 배치
    colL, colM, colR = st.columns([1,1,1], gap="small")

    # 좌·우 : 원본 전체
    colL.image(disp1, caption="Before (original)", width=disp_width)
    colR.image(disp2, caption="After  (original)", width=disp_width)

    # 중앙 : 슬라이더 비교(마스크만) — 크기 동일
    with colM:
        from streamlit_image_comparison import image_comparison
        image_comparison(
            mask1_p, mask2_p,
            label1="Before ROI", label2="After ROI",
            width=disp_width,      # 숫자 필수!
            starting_position=50
        )

# ───────── Main ───────────────────────────────────────────────────────
im = Image.open("favicon.png")
st.set_page_config(page_title="AIOBIO WI Toolkit", page_icon=im, layout="wide")
mode = st.sidebar.radio("Mode", ["Adjust", "Compare"], index=0)
st.title("🦷 AIOBIO WI Toolkit – " + ("Adjust" if mode=="Adjust" else "Compare"))

if mode == "Adjust":
    run_adjust()
else:
    run_compare()
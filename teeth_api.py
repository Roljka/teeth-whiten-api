import io
import math
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# ---------- MediaPipe setup ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# Iekšējās lūpas landmarķi (468 shēmā)
INNER_LIP_IDX = np.array([
    78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,
    87,178,88,95
], dtype=np.int32)

# ---------- Palīgfunkcijas ----------
def _landmarks_to_xy(landmarks, w, h, idx_list):
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)

def _smooth_mask(mask, k=11):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(mask, (k, k), 0)

def _build_mouth_mask(img_bgr, landmarks):
    h, w = img_bgr.shape[:2]

    # iekšējās lūpas (mutes atvēruma) punkti
    inner = _landmarks_to_xy(landmarks, w, h, INNER_LIP_IDX)

    # ja poligons dīvains – atmet
    area = cv2.contourArea(inner)
    if area < 500:
        return np.zeros((h, w), dtype=np.uint8)

    # pamatmaska: tikai mutes iekšpuse
    base = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(base, [inner], 255)

    # Neliels “platuma” pastiepums, lai aizsniegtu līdz pēdējiem zobiem,
    # bet NECELTOS augšup uz smaganām/lūpām
    # → dilatējam ļoti pieticīgi, un pēc tam *erodējam* no augšas (gum guard)
    dil = max(6, int(math.sqrt(area) * 0.03))      # bija agresīvāk
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
    mask = cv2.dilate(base, kernel, iterations=1)

    # “Gum guard”: apgriežam augšējo malu par pāris pikseļiem,
    # lai maska nelien smaganās. Nestrādājam ar vert. shift –
    # griežam ar masīvu, tāpēc nav halo.
    guard = max(4, dil // 2)                       # samazini, ja vēl lien smaganās
    strip = np.zeros_like(mask)
    cv2.rectangle(strip, (0, 0), (w, guard), 255, -1)  # horizontāla josla augšā
    # Pārnesam joslu uz mutes poligona lokālo bbox, lai nesagraizītu citur:
    x,y,ww,hh = cv2.boundingRect(inner)
    strip2 = np.zeros_like(mask)
    y1 = max(y-guard, 0); y2 = y
    strip2[y1:y2, x:x+ww] = 255
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(strip2))

    # Mīksta mala (nekrīt ārā vizuāli)
    mask = _smooth_mask(mask, 17)
    return mask


def _build_teeth_mask(img_bgr, mouth_mask):
    if mouth_mask.sum() == 0:
        return np.zeros_like(mouth_mask)

    h, w = mouth_mask.shape
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    m = mouth_mask > 0
    # --- adaptīvi sliekšņi tikai mutes zonai (stabili arī sliktā gaismā) ---
    L_roi = L[m]; B_roi = B[m]
    L_thr, _ = cv2.threshold(L_roi.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    B_thr, _ = cv2.threshold(B_roi.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # kandidāti: gaiši un nedzelteni
    cand = ((L > L_thr) & (B < B_thr) & m)

    # gum-cut (smaganas/lūpas): sarkans/rozā un/vai liels A + pietiekama S
    gum_red1 = ((H < 15) | (H > 170)) & (S > 40)
    gum_red2 = (A > 148) & (S > 30)
    gums = (gum_red1 | gum_red2)

    # tumšais tukšums (mute iekšā, ēna/tumsa) – zobiem jābūt vismaz vidēji gaišiem
    dark_void = (L < max(80, int(L_thr) - 10))

    # lip-guard atkārtota drošība – vēl šaurāka mala
    guard = 3
    inner_safe = cv2.erode(mouth_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard*2+1, guard*2+1)), 1) > 0

    raw = cand & (~gums) & (~dark_void) & inner_safe

    raw = (raw.astype(np.uint8) * 255)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)

    # aizpildām caurumus, lai nav plankumi
    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    for c in cnts:
        if cv2.contourArea(c) > 80:
            cv2.drawContours(filled, [c], -1, 255, -1)

    teeth_mask = _smooth_mask(filled, 13)

    # kvalitātes kontrole – ja par maz pārklājuma, drošais fallback (mute bez smaganām)
    mouth_area = mouth_mask.sum() / 255.0
    teeth_area = teeth_mask.sum() / 255.0
    if teeth_area / max(mouth_area, 1.0) < 0.30:
        base = cv2.bitwise_and(mouth_mask, cv2.bitwise_not(gums.astype(np.uint8)*255))
        base = cv2.erode(base, np.ones((3,3), np.uint8), 1)
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
        teeth_mask = _smooth_mask(base, 13)

    return teeth_mask

def _teeth_whiten(img_bgr):
    h, w = img_bgr.shape[:2]

    # 1) FaceMesh
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return img_bgr

    landmarks = res.multi_face_landmarks[0].landmark

    # 2) Mutes maska (tava jaunākā _build_mouth_mask versija)
    mouth_mask = _build_mouth_mask(img_bgr, landmarks)
    if np.sum(mouth_mask) < 300:
        return img_bgr

    # 3) ROI
    ys, xs = np.where(mouth_mask > 0)
    y0, y1 = np.clip(ys.min(), 0, h-1), np.clip(ys.max()+1, 0, h)
    x0, x1 = np.clip(xs.min(), 0, w-1), np.clip(xs.max()+1, 0, w)
    roi = img_bgr[y0:y1, x0:x1].copy()
    roi_mm = mouth_mask[y0:y1, x0:x1]

    # 4) Apgaismojuma normalizācija
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    Ln = clahe.apply(L)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # ---------- Pakāpe A: ēnai drošs “sliekšņa” masks ----------
    # “Balts zobs” ≈ gaišs Ln, nav rozā (A zems) un nav dzeltenīgs (B zems), nav ļoti košs (S zems)
    mA = (Ln > 148) & (A < 150) & (B < 165) & (S < 120)
    mA = (mA.astype(np.uint8) * 255)
    mA = cv2.bitwise_and(mA, roi_mm)  # turamies mutes iekšienē
    mA = cv2.morphologyEx(mA, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    mA = _smooth_mask(mA, 9)

    mask = mA
    if np.sum(mask) < 1000:
        # ---------- Pakāpe B: K-means uz mutes pikseļiem ----------
        yy, xx = np.where(roi_mm > 0)
        if len(yy) > 0:
            pts_LAB = np.stack([L[yy, xx], A[yy, xx], B[yy, xx]], axis=1).astype(np.float32)
            # K=2: zobi vs. pārējais; ja ļoti dažāds tonis, K=3 dod bieži labāku dalījumu
            K = 2
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            compactness, labels, centers = cv2.kmeans(pts_LAB, K, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
            centers = centers.reshape(-1,3)  # [K, (L,A,B)]

            # izvēlamies klasi ar visaugstāko L un zemāko B (Gaišs, ne-dzeltens)
            score = centers[:,0] - 0.6*centers[:,2] - 0.2*np.abs(centers[:,1]-128)
            best = int(np.argmax(score))
            mB = np.zeros_like(roi_mm)
            mB[yy, xx] = (labels.ravel()==best).astype(np.uint8)*255

            # gum-guard: izmetam rozīgās zonas
            gums = (A > 145).astype(np.uint8)*255
            mB = cv2.bitwise_and(mB, cv2.bitwise_not(gums))
            mB = cv2.morphologyEx(mB, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
            mB = cv2.morphologyEx(mB, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
            mB = _smooth_mask(mB, 9)

            mask = mB

    if np.sum(mask) < 1000:
        # ---------- Pakāpe C: drošais variants ----------
        # ja joprojām “nekas nenotiek”, balinām visu mutes iekšpusi, BET
        # ar stingru gum-guard, lai nelien smaganās
        gums = (A > 145).astype(np.uint8)*255
        safe = cv2.bitwise_and(roi_mm, cv2.bitwise_not(gums))
        safe = cv2.erode(safe, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)
        safe = _smooth_mask(safe, 7)
        mask = safe

    if np.sum(mask) < 300:
        # ja pilnīgi nav ko balināt – atpakaļ oriģināls
        return img_bgr

    # 5) Balināšana tikai maskā (LAB)
    Lf, Af, Bf = Ln.astype(np.float32), A.astype(np.float32), B.astype(np.float32)
    m = (mask > 0)

    # Gaišums + anti-yellow
    Lf[m] = np.clip(Lf[m]*1.18 + 12, 0, 255)
    Bf[m] = np.clip(Bf[m]*0.78 - 10, 0, 255)

    lab2 = cv2.merge([Lf.astype(np.uint8), Af.astype(np.uint8), Bf.astype(np.uint8)])
    roi_out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Nedaudz izlīdzinām tieši zobu zonu
    blur = cv2.bilateralFilter(roi_out, d=7, sigmaColor=40, sigmaSpace=40)
    roi_out[m] = blur[m]

    # 6) Uzliekam atpakaļ
    out = img_bgr.copy()
    out[y0:y1, x0:x1] = np.where(mask[...,None]>0, roi_out, roi)
    return out

def _read_image_from_request():
    # pieņem 'file' VAI 'image'
    if 'file' in request.files:
        f = request.files['file']
    elif 'image' in request.files:
        f = request.files['image']
    else:
        return None, ("missing file field 'file' (multipart/form-data)", 400)

    try:
        pil = Image.open(f.stream)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:
        return None, (f"cannot open image: {e}", 400)

    img = np.array(pil)[:, :, ::-1]  # RGB->BGR
    return img, None

# ---------- API ----------
@app.route("/whiten", methods=["POST"])
def whiten():
    img_bgr, err = _read_image_from_request()
    if err:
        return jsonify({"error": err[0]}), err[1]

    out_bgr = _teeth_whiten(img_bgr)

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(out_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)

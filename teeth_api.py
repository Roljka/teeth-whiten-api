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
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return img_bgr

    landmarks = res.multi_face_landmarks[0].landmark
    mouth_mask = _build_mouth_mask(img_bgr, landmarks)
    if np.sum(mouth_mask) == 0:
        return img_bgr

    # Strādājam tikai ROI (mutes apgabals) – tas dod stabilitāti tumšā gaismā
    x,y,ww,hh = cv2.boundingRect(np.column_stack(np.where(mouth_mask>0)))
    roi = img_bgr[y:y+hh, x:x+ww].copy()

    # Normalizējam apgaismojumu ar CLAHE (uz L kanāla)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Ln = clahe.apply(L)

    # Sēklas “baltiem, ne-sarkanīgiem” pikseļiem:
    # – liels Ln (gaišs), zems A (ne rozā), zems B (ne dzeltens), zems S (ne ļoti košs)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)

    seed = (Ln > 165) & (A < 140) & (B < 150) & (S < 80)

    # Region grow – 3-kanālu LAB tolerances flood-fill ONLY mutes ROI
    # (tādējādi ēnas tiek “paņemtas” kopā ar blakus zobu toņiem)
    seeds = np.transpose(np.nonzero(seed))
    grown = np.zeros(Ln.shape, np.uint8)
    if len(seeds) > 0:
        # tolerances – jūtība pret ēnām: jo lielāks, jo plašāk “slīd”
        tL, tA, tB = 28, 10, 16
        for sy, sx in seeds[::max(1, len(seeds)//400)]:  # retinām sēklas ātrdarbībai
            refL, refA, refB = int(Ln[sy, sx]), int(A[sy, sx]), int(B[sy, sx])
            diffL = cv2.absdiff(Ln,   np.full(Ln.shape, refL, np.uint8))
            diffA = cv2.absdiff(A,    np.full(A.shape,  refA, np.uint8))
            diffB = cv2.absdiff(B,    np.full(B.shape,  refB, np.uint8))
            region = (diffL <= tL) & (diffA <= tA) & (diffB <= tB)
            grown |= region.astype(np.uint8)*255

    # Piesienam mutes masai un iztīrām smaganas (rozīgais A-kanāls)
    grown = cv2.bitwise_and(grown, grown, mask=mouth_mask[y:y+hh, x:x+ww])
    gums = (A > 145).astype(np.uint8)*255
    grown = cv2.bitwise_and(grown, cv2.bitwise_not(gums))

    # Aizlāpām caurumus + nolīdzinām malas (bez izliešanas uz lūpām)
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), 1)
    grown = _smooth_mask(grown, 11)

    # Ja kaut kādu iemeslu dēļ zobi nav atrasti – atgriež oriģinālu
    if np.sum(grown) < 500:
        return img_bgr

    # BALINĀŠANA tikai “grown” pikseļos
    Lf, Af, Bf = Ln.astype(np.float32), A.astype(np.float32), B.astype(np.float32)
    m = (grown > 0)

    # gaišums + dzeltenuma samazināšana
    Lf[m] = np.clip(Lf[m]*1.18 + 10, 0, 255)
    Bf[m] = np.clip(Bf[m]*0.78 - 10, 0, 255)

    L2 = Lf.astype(np.uint8); A2 = Af.astype(np.uint8); B2 = Bf.astype(np.uint8)
    lab2 = cv2.merge([L2, A2, B2])

    roi_out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Mazs bilaterālais blur tieši zobu iekšienē – pret plankumiem
    blur = cv2.bilateralFilter(roi_out, d=7, sigmaColor=40, sigmaSpace=40)
    roi_out[m] = blur[m]

    # Uzliekam atpakaļ tikai zobu zonu
    out = img_bgr.copy()
    mask_full = np.zeros((h, w), np.uint8)
    mask_full[y:y+hh, x:x+ww] = grown
    out[y:y+hh, x:x+ww] = np.where(mask_full[y:y+hh, x:x+ww, None]>0, roi_out, roi)

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

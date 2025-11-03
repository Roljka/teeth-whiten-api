import io
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# ------------ MediaPipe ------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)

# Mutes iekšējais/ārējais gredzens (FaceMesh topo)
MOUTH_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
MOUTH_INNER = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]

# ------------ Palīgfunkcijas ------------
def load_image_fix_orientation(file_storage, max_side=1600):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    rgb = img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def enhance_for_detection(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

def poly_from_landmarks(h, w, lms, idxs):
    pts = []
    for i in idxs:
        lm = lms[i]
        pts.append([int(lm.x*w), int(lm.y*h)])
    return np.array(pts, np.int32)

def mask_from_poly(h, w, pts):
    m = np.zeros((h,w), np.uint8)
    if pts.size >= 6:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(m, hull, 255)
    return m

def build_floor(mask):
    h,w = mask.shape
    floor = np.full(w, -1, np.int32)
    ys, xs = np.where(mask>0)
    for x in range(w):
        col = ys[xs==x]
        if col.size: floor[x] = col.max()
    return floor

def build_ceiling(mask):
    h,w = mask.shape
    ceil = np.full(w, -1, np.int32)
    ys, xs = np.where(mask>0)
    for x in range(w):
        col = ys[xs==x]
        if col.size: ceil[x] = col.min()
    return ceil

def cut_below(mask, floor, lift_px):
    h,w = mask.shape
    out = mask.copy()
    for x in range(w):
        y = floor[x]
        if y >= 0:
            out[y+lift_px:h, x] = 0
    return out

def cut_above(mask, ceiling, shave_px):
    h,w = mask.shape
    out = mask.copy()
    for x in range(w):
        y = ceiling[x]
        if y >= 0:
            out[0:max(0, y+shave_px), x] = 0
    return out

# --- stingrāka smaganu maska (HSV sarkanie + LAB A>pink) + top josla
def gums_mask(bgr, mouth_mask, top_guard_px=3, loose=False):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _,A,_ = cv2.split(lab)

    if not loose:
        red = (((H <= 12) | (H >= 170)) & (S > 30))
        pink = (A > 155)
    else:
        red = (((H <= 14) | (H >= 168)) & (S > 25))
        pink = (A > 150)

    gum_color = ((red | pink) & (mouth_mask>0))

    # top josla – smaganas mutes “griestos”
    ceil = build_ceiling(mouth_mask)
    top_band = np.zeros_like(mouth_mask)
    h,w = mouth_mask.shape
    for x in range(w):
        y = ceil[x]
        if y >= 0:
            y2 = min(h, y + top_guard_px)
            top_band[y:y2, x] = 255

    gum = np.zeros_like(mouth_mask)
    gum[ gum_color | (top_band>0) ] = 255

    gum = cv2.morphologyEx(gum, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)
    return gum

# paplašina zobu kolonnas vertikāli iekš drošās mutes
def expand_cols_vertical(teeth, safe_mouth, up=2, down=3):
    h,w = teeth.shape
    out = teeth.copy()
    xs = np.where(safe_mouth.sum(0) > 0)[0]
    for x in xs:
        ys = np.where(teeth[:,x] > 0)[0]
        if ys.size >= 2:
            y1, y2 = ys.min(), ys.max()
            y1 = max(0, y1 - up)
            y2 = min(h-1, y2 + down)
            col_safe = safe_mouth[:,x] > 0
            out[y1:y2+1, x] = np.where(col_safe[y1:y2+1], 255, out[y1:y2+1, x])
    # izlīdzinām
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,5)), 1)
    return out

# aizpilda pa rindām (platums)
def fill_rows(teeth, mouth):
    h,w = teeth.shape
    out = teeth.copy()
    ys = np.where(mouth.sum(1) > 0)[0]
    for y in ys:
        xs = np.where(teeth[y] > 0)[0]
        if xs.size >= 2:
            x1, x2 = xs.min(), xs.max()
            row_mouth = mouth[y] > 0
            out[y, x1:x2+1] = np.where(row_mouth[x1:x2+1], 255, out[y, x1:x2+1])
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return out

# ------------ Masku veidotāji ------------
def build_teeth_mask(bgr, lms, lowlight=False):
    h,w = bgr.shape[:2]
    outer = mask_from_poly(h, w, poly_from_landmarks(h,w,lms,MOUTH_OUTER))
    inner = mask_from_poly(h, w, poly_from_landmarks(h,w,lms,MOUTH_INNER))

    # plašums – lai aizsniedz sānu zobus
    # vertikālais kodols lielāks (9→11), horizontāli nedaudz platāks (33→37)
    inner_wide = cv2.dilate(inner, cv2.getStructuringElement(cv2.MORPH_RECT,(37,11)), 1)

    # lūpu maska (ārējais – iekšējais)
    lips_only = cv2.subtract(outer, cv2.dilate(inner, None, 1))

    floor = build_floor(outer)
    ceil  = build_ceiling(outer)

    # nogriežam apakšlūpu 1px, augšā drošības josla 3–4px
    inner_wide = cut_below(inner_wide, floor, lift_px=1)
    inner_wide = cut_above(inner_wide, ceil, shave_px=(4 if lowlight else 3))

    # drošā mute – kur drīkst balināt (bez lūpām/bez smaganām)
    gum = gums_mask(bgr, inner_wide, top_guard_px=(4 if lowlight else 3), loose=lowlight)
    safe_mouth = cv2.subtract(inner_wide, gum)
    safe_mouth = cv2.subtract(safe_mouth, lips_only)

    # sākotnējā zobu semantika: zemi S, augsts V, zems A (ne pink)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); H,S,V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); L,A,B = cv2.split(lab)
    if not lowlight:
        candidates = ((S < 90) & (V > 145) & (A < 150) & (safe_mouth > 0))
    else:
        candidates = ((S < 110) & (V > 120) & (A < 155) & (safe_mouth > 0))

    teeth = np.zeros((h,w), np.uint8); teeth[candidates] = 255
    # tīrīšana
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)

    # aizpildām pa rindām (platums) un paplašinām pa kolonnām (augstums)
    teeth = fill_rows(teeth, safe_mouth)
    teeth = expand_cols_vertical(teeth, safe_mouth, up=2, down=3)

    # pēdējais klips drošajā mutē (nekad neiet smaganās/lūpā)
    teeth = cv2.bitwise_and(teeth, safe_mouth)

    return teeth, inner_wide, lips_only, safe_mouth

# ------------ Balināšana ------------
def whiten_only_teeth(bgr, mask, l_gain=10, b_shift=18):
    if np.count_nonzero(mask) == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L,A,B = cv2.split(lab)
    m = mask > 0
    Ln = L.astype(np.int16); Bn = B.astype(np.int16)
    Ln[m] = np.clip(Ln[m] + l_gain, 0, 255)
    Bn[m] = np.clip(Bn[m] - b_shift, 0, 255)  # mazāks, lai nebūtu zilgans
    out = cv2.cvtColor(cv2.merge([Ln.astype(np.uint8), A, Bn.astype(np.uint8)]),
                       cv2.COLOR_LAB2BGR)
    return out

# ------------ API ------------
@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing: use multipart/form-data 'file'"), 400

        bgr = load_image_fix_orientation(request.files["file"])
        det = enhance_for_detection(bgr)

        res = face_mesh.process(cv2.cvtColor(det, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422
        lms = res.multi_face_landmarks[0].landmark

        # 1) parastā maska
        teeth, inner_wide, lips_only, safe_mouth = build_teeth_mask(det, lms, lowlight=False)

        # ja pārklājums par zemu, pārslēdzam “lowlight” režīmu (mīkstāki sliekšņi + lielāks gum guard)
        covered = np.count_nonzero(teeth)
        mouth_area = np.count_nonzero(safe_mouth)
        if mouth_area == 0 or (mouth_area > 0 and covered/mouth_area < 0.38):
            teeth, _, _, _ = build_teeth_mask(det, lms, lowlight=True)

        out = whiten_only_teeth(bgr, teeth, l_gain=10, b_shift=18)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return jsonify(error="Encode failed"), 500
        return send_file(io.BytesIO(buf.tobytes()),
                         mimetype="image/jpeg",
                         as_attachment=False,
                         download_name="whitened.jpg")
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

import io, os, cv2, numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                  refine_landmarks=False, min_detection_confidence=0.5)

MOUTH_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
MOUTH_INNER = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]

# ---------- utils ----------
def load_image_fix_orientation(file_storage, max_side=1600):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    w,h = img.size; s = min(1.0, max_side/max(w,h))
    if s<1.0: img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def enhance_for_detection(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0,(8,8)); L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2,A,B]), cv2.COLOR_LAB2BGR)

def poly_from_landmarks(h,w,lms,idxs):
    return np.array([[int(lms[i].x*w), int(lms[i].y*h)] for i in idxs], np.int32)

def mask_from_poly(h,w,pts):
    m = np.zeros((h,w), np.uint8)
    if pts.size>=6:
        hull = cv2.convexHull(pts); cv2.fillConvexPoly(m,hull,255)
    return m

def build_floor(mask):
    h,w = mask.shape; floor = np.full(w,-1,np.int32)
    ys,xs = np.where(mask>0)
    for x in range(w):
        col = ys[xs==x];  floor[x] = col.max() if col.size else -1
    return floor

def build_ceiling(mask):
    h,w = mask.shape; ceil = np.full(w,-1,np.int32)
    ys,xs = np.where(mask>0)
    for x in range(w):
        col = ys[xs==x];  ceil[x] = col.min() if col.size else -1
    return ceil

def cut_below(mask, floor, lift_px):
    """Nogriez visu ZEM mutes grīdas (grīda + lift_px..h)."""
    out = mask.copy()
    h, w = out.shape
    for x in range(w):
        y = floor[x]
        if y >= 0:
            y1 = min(h, y + lift_px)
            if y1 < h:
                out[y1:h, x] = 0
    return out

def cut_above(mask, ceiling, shave_px):
    """Nogriez visu VIRS mutes griestiem (0..griesti+shave_px)."""
    out = mask.copy()
    h, w = out.shape
    for x in range(w):
        y = ceiling[x]
        if y >= 0:
            y2 = max(0, min(h, y + shave_px))
            if y2 > 0:
                out[0:y2, x] = 0
    return out

def gums_mask(bgr, mouth_mask, top_guard_px=3, loose=False):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); H,S,V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); _,A,_ = cv2.split(lab)
    if not loose:
        red = (((H<=12)|(H>=170)) & (S>30)); pink = (A>155)
    else:
        red = (((H<=14)|(H>=168)) & (S>25)); pink = (A>150)
    gum_color = ((red|pink) & (mouth_mask>0))

    ceil = build_ceiling(mouth_mask)
    top_band = np.zeros_like(mouth_mask)
    h,w = mouth_mask.shape
    for x in range(w):
        y=ceil[x]
        if y>=0: top_band[y:min(h,y+top_guard_px),x]=255

    gum = np.zeros_like(mouth_mask)
    gum[ gum_color | (top_band>0) ] = 255
    gum = cv2.morphologyEx(gum, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    return gum

def fill_rows(teeth, mouth):
    h,w = teeth.shape; out = teeth.copy()
    for y in np.where(mouth.sum(1)>0)[0]:
        xs = np.where(teeth[y]>0)[0]
        if xs.size>=2:
            x1,x2 = xs.min(), xs.max()
            row_ok = mouth[y]>0
            out[y, x1:x2+1] = np.where(row_ok[x1:x2+1], 255, out[y, x1:x2+1])
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    return out

def expand_cols_vertical(teeth, safe_mouth, up=3, down=4):
    h,w = teeth.shape; out = teeth.copy()
    for x in np.where(safe_mouth.sum(0)>0)[0]:
        ys = np.where(teeth[:,x]>0)[0]
        if ys.size>=2:
            y1,y2 = ys.min(), ys.max()
            y1=max(0,y1-up); y2=min(h-1,y2+down)
            col_ok = safe_mouth[:,x]>0
            out[y1:y2+1, x] = np.where(col_ok[y1:y2+1],255,out[y1:y2+1,x])
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,5)),1)
    return out

# ---------- mask builders + fallbacks ----------
def build_masks(bgr, lms, lowlight=False):
    h,w = bgr.shape[:2]
    outer = mask_from_poly(h,w, poly_from_landmarks(h,w,lms,MOUTH_OUTER))
    inner = mask_from_poly(h,w, poly_from_landmarks(h,w,lms,MOUTH_INNER))

    inner_wide = cv2.dilate(inner, cv2.getStructuringElement(cv2.MORPH_RECT,(37,11)),1)
    lips_only  = cv2.subtract(outer, cv2.dilate(inner, None, 1))

    floor = build_floor(outer); ceil = build_ceiling(outer)
    inner_wide = cut_below(inner_wide, floor, lift_px=1)
    inner_wide = cut_above(inner_wide, ceil, shave_px=(4 if lowlight else 3))

    gum = gums_mask(bgr, inner_wide, top_guard_px=(4 if lowlight else 3), loose=lowlight)
    safe_mouth = cv2.subtract(inner_wide, gum)
    safe_mouth = cv2.subtract(safe_mouth, lips_only)
    return outer, inner_wide, lips_only, safe_mouth

def teeth_from_colors(bgr, safe_mouth, lowlight=False):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); H,S,V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); L,A,B = cv2.split(lab)
    if not lowlight:
        cand = ((S<90)&(V>145)&(A<150)&(safe_mouth>0))
    else:
        cand = ((S<110)&(V>120)&(A<155)&(safe_mouth>0))
    m = np.zeros_like(safe_mouth); m[cand]=255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    return m

def teeth_adaptive_percentiles(bgr, safe_mouth):
    # procenti no mutes iekšpuses histogrammām
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); H,S,V = cv2.split(hsv)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); _,A,_ = cv2.split(lab)
    m = safe_mouth>0
    if np.count_nonzero(m)==0: return np.zeros_like(safe_mouth)
    s_th = np.percentile(S[m], 60)
    v_th = np.percentile(V[m], 55)
    a_th = np.percentile(A[m], 65)
    cand = ((S < s_th) & (V > v_th) & (A < a_th) & m)
    out = np.zeros_like(safe_mouth); out[cand]=255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    return out

def teeth_adaptive_contrast(bgr, safe_mouth):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); V = hsv[:,:,2]
    clahe = cv2.createCLAHE(2.0,(8,8)); v2 = clahe.apply(V)
    # Otsu tikai mutes iekšpusē
    v_masked = np.where(safe_mouth>0, v2, 0).astype(np.uint8)
    # izvelkam slieksni no pikseļiem mutes iekšpusē
    vals = v_masked[v_masked>0]
    if vals.size<50:
        return np.zeros_like(safe_mouth)
    _,th = cv2.threshold(v_masked, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.bitwise_and(th, safe_mouth)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),1)
    return th

def make_teeth_mask(bgr, lms):
    # parastais
    outer, inner_wide, lips_only, safe = build_masks(bgr, lms, lowlight=False)
    t = teeth_from_colors(bgr, safe, lowlight=False)
    t = fill_rows(t, safe); t = expand_cols_vertical(t, safe, up=3, down=4)
    t = cv2.bitwise_and(t, safe)
    cov = np.count_nonzero(t); area = np.count_nonzero(safe)
    if area>0 and cov/area >= 0.38:
        return t

    # lowlight
    outer, inner_wide, lips_only, safe = build_masks(bgr, lms, lowlight=True)
    t = teeth_from_colors(bgr, safe, lowlight=True)
    t = fill_rows(t, safe); t = expand_cols_vertical(t, safe, up=3, down=4)
    t = cv2.bitwise_and(t, safe)
    cov = np.count_nonzero(t); area = np.count_nonzero(safe)
    if area>0 and cov/area >= 0.35:
        return t

    # adaptīvie procenti
    t = teeth_adaptive_percentiles(bgr, safe)
    t = fill_rows(t, safe); t = expand_cols_vertical(t, safe, up=3, down=4)
    t = cv2.bitwise_and(t, safe)
    cov = np.count_nonzero(t)
    if area>0 and cov/area >= 0.33:
        return t

    # lokālais kontrasts
    t = teeth_adaptive_contrast(bgr, safe)
    t = fill_rows(t, safe); t = expand_cols_vertical(t, safe, up=3, down=4)
    t = cv2.bitwise_and(t, safe)
    cov = np.count_nonzero(t)
    if area>0 and cov/area >= 0.30:
        return t

    # pēdējais glābiņš — viss safe_mouth, nogriežam 1px augša/apakša
    ceil = build_ceiling(safe); floor = build_floor(safe)
    t = safe.copy()
    t = cut_above(t, ceil, shave_px=1)
    t = cut_below(t, floor, lift_px=1)
    return t

# ---------- whitening ----------
def whiten_only_teeth(bgr, mask, l_gain=10, b_shift=18):
    if np.count_nonzero(mask)==0: return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); L,A,B = cv2.split(lab)
    m = mask>0; Ln = L.astype(np.int16); Bn = B.astype(np.int16)
    Ln[m] = np.clip(Ln[m]+l_gain, 0, 255)
    Bn[m] = np.clip(Bn[m]-b_shift, 0, 255)
    return cv2.cvtColor(cv2.merge([Ln.astype(np.uint8),A,Bn.astype(np.uint8)]), cv2.COLOR_LAB2BGR)

# ---------- API ----------
@app.route("/health")
def health(): return jsonify(ok=True)

@app.route("/whiten", methods=["POST"])
def whiten():
    try:
        if "file" not in request.files:
            return jsonify(error="File missing (use multipart/form-data 'file')"), 400
        bgr = load_image_fix_orientation(request.files["file"])
        det = enhance_for_detection(bgr)
        res = face_mesh.process(cv2.cvtColor(det, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return jsonify(error="Face not found"), 422
        lms = res.multi_face_landmarks[0].landmark

        teeth_mask = make_teeth_mask(det, lms)
        out = whiten_only_teeth(bgr, teeth_mask, l_gain=10, b_shift=18)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok: return jsonify(error="Encode failed"), 500
        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg",
                         as_attachment=False, download_name="whitened.jpg")
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

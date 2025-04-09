# verify_face.py (FastAPI용)

import json
import os
import numpy as np
import cv2
import base64
from keras_facenet import FaceNet
from mtcnn import MTCNN

# 설정
EMBEDDING_DIR = "embeddings"
DB_FILE = os.path.join(EMBEDDING_DIR, "face_embeddings.json")
THRESHOLD = 0.7  # 거리 기준

embedder = FaceNet()
detector = MTCNN()

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def verify_face(image_base64: str):
    try:
        if not os.path.exists(DB_FILE):
            return { "message": "등록된 사용자 데이터가 없습니다." }

        with open(DB_FILE, "r") as f:
            face_db = json.load(f)

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces = detector.detect_faces(frame)
        if len(faces) == 0:
            return { "message": "감지된 얼굴 없음" }

        x, y, w, h = faces[0]['box']
        face = frame[y:y+h, x:x+w]
        input_embedding = embedder.embeddings([face])[0]

        best_match = None
        best_dist = float("inf")
        for user_id, embeddings in face_db.items():
            for stored_embedding in embeddings:
                dist = euclidean_distance(input_embedding, stored_embedding)
                print(f"[🔍 {user_id}] 거리: {dist:.3f}")
                if dist < best_dist:
                    best_dist = dist
                    best_match = user_id

        if best_dist < THRESHOLD:
            return { "user_id": best_match }
        else:
            return { "message": "인증 실패" }

    except Exception as e:
        print(f"[❌ 오류] {e}")
        return { "message": f"서버 오류: {str(e)}" }

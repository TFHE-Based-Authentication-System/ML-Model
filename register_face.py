import cv2
import json
import os
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

# 경로 설정
EMBEDDING_DIR = "embeddings"
DB_FILE = os.path.join(EMBEDDING_DIR, "face_embeddings.json")
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# 모델 로드
embedder = FaceNet()
detector = MTCNN()

# 벡터 저장 함수
def save_embedding(user_id, embedding):
    # DB 로드
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            face_db = json.load(f)
    else:
        face_db = {}

    # 기존 사용자면 벡터 리스트에 추가, 아니면 새 리스트 생성
    if user_id in face_db:
        face_db[user_id].append(embedding)
    else:
        face_db[user_id] = [embedding]

    # 저장
    with open(DB_FILE, "w") as f:
        json.dump(face_db, f, indent=4)
    print(f"[✅ 등록 성공] {user_id}의 새로운 벡터가 추가 저장되었습니다.")

# 얼굴 등록 함수
# FastAPI용 얼굴 등록 함수 (base64 이미지와 user_id를 받아 처리)
def register_face(image_bytes, user_id):
    try:
        # 이미지 디코딩
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 얼굴 검출
        faces = detector.detect_faces(frame)
        if len(faces) == 0:
            return {"success": False, "message": "감지된 얼굴이 없습니다."}

        x, y, w, h = faces[0]['box']
        face = frame[y:y+h, x:x+w]

        # 임베딩 추출 및 저장
        embedding = embedder.embeddings([face])[0].tolist()
        save_embedding(user_id, embedding)

        return {"success": True, "message": f"{user_id} 등록 완료."}

    except Exception as e:
        print(f"[❌ 오류] {e}")
        return {"success": False, "message": f"에러 발생: {str(e)}"}
    
if __name__ == "__main__":
    register_face()

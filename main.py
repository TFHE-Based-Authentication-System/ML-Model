from fastapi import FastAPI
from pydantic import BaseModel
import base64
from register_face import register_face  # 너가 만든 얼굴 등록 함수
from verify_face import verify_face

app = FastAPI()

class ImageRequest(BaseModel):
    image_base64: str
    user_id: str
class VerifyRequest(BaseModel):
    image_base64: str

@app.post("/api/image/register")
async def register(data: ImageRequest):
    image_bytes = base64.b64decode(data.image_base64)
    result = register_face(image_bytes, data.user_id)  # ✅ 인자 2개 넘겨줘야 함
    return {"result": result}


@app.post("/api/image/verify")
async def verify(data: VerifyRequest):
    result = verify_face(data.image_base64)
    return result
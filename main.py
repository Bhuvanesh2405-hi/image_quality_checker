from fastapi import FastAPI, UploadFile, File
from predictor import check_clarity
import uvicorn
app = FastAPI(title="Image Clarity API")
@app.post("/check-clarity")
async def clarity_api(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = check_clarity(img_bytes)
    if result == "clear":
        return {
            "status": "success",
            "clarity": "clear",
            "reason": "image is sharp"
        }
    else:
        return {
            "status": "fail",
            "clarity": "not_clear",
            "reason": "image is blurry"
        }
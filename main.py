from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from predictor import check_clarity, is_image_clear
import uvicorn

app = FastAPI(title="Image Clarity API")


@app.post("/check-clarity")
async def clarity_api(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = check_clarity(img_bytes)

    if result["verdict"] == "clear":
        return JSONResponse(
            status_code=200,
            content={
                "status"   : "success",
                "clarity"  : "clear",
                "warnings" : result["warnings"],   # soft issues, if any
                "metrics"  : result["metrics"],
            }
        )
    else:
        return JSONResponse(
            status_code=422,
            content={
                "status"     : "fail",
                "clarity"    : "not_clear",
                "hard_fails" : result["hard_fails"],
                "warnings"   : result["warnings"],
                "metrics"    : result["metrics"],
            }
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

# modules/response_builder.py
from fastapi.responses import JSONResponse

def success_response(data=None, message="성공", status_code=200):
    content = {
        "status": "success",
        "message": message,
        "data": data if data is not None else {}
    }
    return JSONResponse(content=content, status_code=status_code)
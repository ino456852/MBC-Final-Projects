from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from core.database import MongoDB
from domains.auth.routes.auth_routes import router as auth_router
from domains.users.routes.user_routes import router as users_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    MongoDB.connect()
    yield
    MongoDB.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 리액트 앱 도메인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )
    
app.include_router(auth_router)
app.include_router(users_router)
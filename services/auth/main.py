"""
Authentication Service — Port 8007
Handles user authentication, RBAC authorization, and JWT issuance.
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict
import uuid
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.schemas import TokenResponse, UserInfo, UserRole, TokenRequest
from shared.config import settings

app = FastAPI(
    title="Authentication Service",
    description="Handles user auth, RBAC, and JWT tokens for the AI Handover Platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Password hashing ──────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["sha256_crypt", "bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ── In-memory user store (replace with DB in production) ─────────────────────
USERS_DB: Dict[str, dict] = {
    "operator1": {
        "user_id":  str(uuid.uuid4()),
        "username": "operator1",
        "hashed_password": pwd_context.hash("operator_pass"),
        "role": UserRole.NETWORK_OPERATOR,
        "email": "operator1@5g-platform.com",
    },
    "scientist1": {
        "user_id":  str(uuid.uuid4()),
        "username": "scientist1",
        "hashed_password": pwd_context.hash("scientist_pass"),
        "role": UserRole.DATA_SCIENTIST,
        "email": "scientist1@5g-platform.com",
    },
    "admin": {
        "user_id":  str(uuid.uuid4()),
        "username": "admin",
        "hashed_password": pwd_context.hash("admin_pass"),
        "role": UserRole.ADMIN,
        "email": "admin@5g-platform.com",
    },
}


# ── JWT helpers ───────────────────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def get_user(username: str) -> Optional[dict]:
    return USERS_DB.get(username)

def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + expires_delta
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Dependency: get current user ──────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInfo:
    # Accept internal service-to-service token
    if token == settings.INTERNAL_SERVICE_TOKEN:
        return UserInfo(
            user_id="internal",
            username="internal-service",
            role=UserRole.ADMIN,
            email="internal@system",
        )
    payload = decode_token(token)
    username: str = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return UserInfo(
        user_id=user["user_id"],
        username=user["username"],
        role=user["role"],
        email=user["email"],
    )

def require_role(*roles: UserRole):
    """Role-based access control dependency factory."""
    async def checker(current_user: UserInfo = Depends(get_current_user)):
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not authorized for this action. "
                       f"Required: {[r.value for r in roles]}",
            )
        return current_user
    return checker


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate and receive a JWT bearer token.

    **Test credentials:**
    - operator1 / operator_pass
    - scientist1 / scientist_pass
    - admin / admin_pass
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expire = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=expire,
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        role=user["role"],
    )


@app.get("/auth/me", response_model=UserInfo, tags=["Authentication"])
async def get_me(current_user: UserInfo = Depends(get_current_user)):
    """Return profile of the currently authenticated user."""
    return current_user


@app.post("/auth/verify", tags=["Authentication"])
async def verify_token(token: str):
    """Validate a JWT token (used by API Gateway for inter-service auth)."""
    payload = decode_token(token)
    return {"valid": True, "username": payload.get("sub"), "role": payload.get("role")}


@app.get("/auth/users", tags=["Admin"])
async def list_users(admin: UserInfo = Depends(require_role(UserRole.ADMIN))):
    """List all users. Admin only."""
    return [
        {"user_id": u["user_id"], "username": u["username"],
         "role": u["role"], "email": u["email"]}
        for u in USERS_DB.values()
    ]


@app.get("/health", tags=["Health"])
async def health():
    return {"service": "auth", "status": "healthy", "timestamp": datetime.utcnow()}

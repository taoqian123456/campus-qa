"""
密码哈希（passlib + bcrypt）与 JWT 生成/解析（python-jose，HS256）。

注意：passlib 与 bcrypt>=4.1 不兼容，venv 里必须固定 bcrypt==4.0.1，
否则哈希密码时报 "AttributeError: module 'bcrypt' has no attribute '__about__'"。
"""
from datetime import datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext

from config import ACCESS_TOKEN_EXPIRE_DAYS, SECRET_KEY

ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """明文密码 -> bcrypt 哈希"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验明文密码与哈希是否匹配"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int) -> str:
    """为指定用户签发 JWT，有效期见 config.ACCESS_TOKEN_EXPIRE_DAYS"""
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> int | None:
    """解析 JWT，返回用户 id；签名无效或已过期返回 None。"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None

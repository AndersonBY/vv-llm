"""
轻量级 GCP Access Token 生成器
"""

import base64
import json
import os
import time
from typing import cast
from pathlib import Path

import httpx2

TOKEN_URI = "https://oauth2.googleapis.com/token"


def _refresh_user_token_with_expiry(creds: dict, client_kwargs: dict) -> tuple[str, int]:
    """刷新用户 token，返回 (token, expires_in)"""
    token_uri = creds.get("token_uri", TOKEN_URI)
    with httpx2.Client(**client_kwargs) as client:
        resp = client.post(
            token_uri,
            data={
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
                "refresh_token": creds["refresh_token"],
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"], data.get("expires_in", 3600)


def _get_sa_token_with_expiry(sa: dict, client_kwargs: dict) -> tuple[str, int]:
    """获取服务账号 token，返回 (token, expires_in)"""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

    token_uri = sa.get("token_uri", TOKEN_URI)

    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    payload = {
        "iss": sa["client_email"],
        "sub": sa["client_email"],
        "aud": token_uri,
        "iat": now,
        "exp": now + 3600,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
    }

    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    h = b64(json.dumps(header).encode())
    p = b64(json.dumps(payload).encode())
    msg = f"{h}.{p}".encode()

    key = serialization.load_pem_private_key(sa["private_key"].encode(), None)
    if not isinstance(key, RSAPrivateKey):
        raise TypeError("服务账号私钥必须是 RSA 密钥")
    sig = key.sign(msg, padding.PKCS1v15(), hashes.SHA256())

    jwt = f"{h}.{p}.{b64(sig)}"

    with httpx2.Client(**client_kwargs) as client:
        resp = client.post(
            token_uri,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"], data.get("expires_in", 3600)


def get_token_with_cache(
    credentials: dict | None = None,
    proxy: str | None = None,
    cached_token: str | None = None,
    cached_expires_at: float | None = None,
    refresh_threshold: int = 300,
) -> tuple[str, float]:
    """
    获取 access token，支持缓存

    如果提供了有效的缓存 token，直接返回；否则刷新并返回新 token 和过期时间。
    适用于外部系统（如 Redis）管理 token 缓存的场景。

    Args:
        credentials: GCP 凭证字典，可以是用户凭证或服务账号
        proxy: HTTP 代理地址
        cached_token: 缓存的 token
        cached_expires_at: 缓存 token 的过期时间戳 (Unix time)
        refresh_threshold: 提前刷新的秒数，默认 300 秒 (5分钟)

    Returns:
        Tuple of (access_token, expires_at_timestamp)

    Example:
        # 从 Redis 读取缓存
        token_data = redis.get(f"gcp_token:{project_id}")
        cached_token = token_data.get('token') if token_data else None
        cached_expires_at = token_data.get('expires_at') if token_data else None

        # 获取 token（如果缓存有效则直接返回，否则刷新）
        token, expires_at = get_token_with_cache(
            credentials=credentials,
            proxy=proxy,
            cached_token=cached_token,
            cached_expires_at=cached_expires_at,
        )

        # 更新 Redis 缓存
        redis.set(f"gcp_token:{project_id}", {'token': token, 'expires_at': expires_at})
    """
    # 检查缓存 token 是否有效
    if cached_token and cached_expires_at:
        if time.time() < cached_expires_at - refresh_threshold:
            return cached_token, cached_expires_at

    # 需要刷新
    client_kwargs = {"proxy": proxy} if proxy else {}

    # 如果没传凭证，从文件加载
    if credentials is None:
        adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if adc_path.exists():
            with open(adc_path) as f:
                credentials = json.load(f)
        else:
            sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if sa_path and Path(sa_path).exists():
                with open(sa_path) as f:
                    credentials = json.load(f)
            else:
                raise RuntimeError("未找到凭证，请先运行: gcloud auth application-default login")

    if "refresh_token" in cast(dict, credentials):
        token, expires_in = _refresh_user_token_with_expiry(credentials, client_kwargs)
    elif "private_key" in cast(dict, credentials):
        token, expires_in = _get_sa_token_with_expiry(credentials, client_kwargs)
    else:
        raise ValueError("无效的凭证格式")

    return token, time.time() + expires_in

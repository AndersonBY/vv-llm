"""
轻量级 GCP Access Token 生成器
"""

import base64
import json
import os
import subprocess
import time
from typing import Any, cast
from pathlib import Path

import httpx

TOKEN_URI = "https://oauth2.googleapis.com/token"


class TokenManager:
    """
    Token 管理器，支持自动刷新过期 token

    用法:
        manager = TokenManager(credentials={"refresh_token": ...}, proxy="http://...")
        token = manager.token  # 自动刷新过期 token
    """

    def __init__(
        self,
        credentials: dict | None = None,
        proxy: str | None = None,
        refresh_threshold: int = 300,  # 提前 5 分钟刷新
    ):
        """
        Args:
            credentials: 凭证字典，None 则自动从文件读取
            proxy: HTTP 代理地址
            refresh_threshold: 提前刷新的秒数，默认 300 秒 (5分钟)
        """
        self._credentials = credentials
        self._proxy = proxy
        self._refresh_threshold = refresh_threshold
        self._token: str | None = None
        self._expires_at: float = 0

        # 如果没传凭证，从文件加载
        if self._credentials is None:
            self._credentials = self._load_credentials()
        self._credentials = cast(dict[str, Any], self._credentials)

    def _load_credentials(self) -> dict:
        """从文件加载凭证"""
        # 优先检查 ADC 文件
        adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if adc_path.exists():
            with open(adc_path) as f:
                return json.load(f)

        # 检查服务账号
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if sa_path and Path(sa_path).exists():
            with open(sa_path) as f:
                return json.load(f)

        raise RuntimeError("未找到凭证，请先运行: gcloud auth application-default login")

    @property
    def expired(self) -> bool:
        """检查 token 是否过期或即将过期"""
        if self._token is None:
            return True
        return time.time() >= (self._expires_at - self._refresh_threshold)

    @property
    def token(self) -> str:
        """获取 token，过期自动刷新"""
        if self.expired:
            self.refresh()
        return cast(str, self._token)

    def refresh(self) -> str:
        """强制刷新 token"""
        client_kwargs = {"proxy": self._proxy} if self._proxy else {}
        credentials = cast(dict[str, Any], self._credentials)

        if "refresh_token" in credentials:
            self._token, expires_in = _refresh_user_token_with_expiry(
                credentials,
                client_kwargs,
            )
        elif "private_key" in credentials:
            self._token, expires_in = _get_sa_token_with_expiry(
                credentials,
                client_kwargs,
            )
        else:
            raise ValueError("无效的凭证格式")

        self._expires_at = time.time() + expires_in
        return self._token


def _refresh_user_token_with_expiry(creds: dict, client_kwargs: dict) -> tuple[str, int]:
    """刷新用户 token，返回 (token, expires_in)"""
    token_uri = creds.get("token_uri", TOKEN_URI)
    with httpx.Client(**client_kwargs) as client:
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

    with httpx.Client(**client_kwargs) as client:
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


def get_token_from_adc(
    credentials: dict | None = None,
    proxy: str | None = None,
) -> str:
    """
    从 Application Default Credentials 获取 access token

    Args:
        credentials: 凭证字典，可以是:
            - 用户凭证 (包含 refresh_token, client_id, client_secret)
            - 服务账号 (包含 private_key, client_email)
            - None: 自动从文件读取
        proxy: HTTP 代理地址，如 "http://127.0.0.1:7890"

    Returns:
        access_token 字符串
    """
    client_kwargs = {"proxy": proxy} if proxy else {}

    # 如果传入了凭证字典
    if credentials is not None:
        if "refresh_token" in credentials:
            return _refresh_user_token(credentials, client_kwargs)
        elif "private_key" in credentials:
            return _get_token_from_service_account_dict(credentials, client_kwargs)
        else:
            raise ValueError("无效的凭证格式")

    # 优先检查 ADC 文件 (gcloud auth application-default login)
    adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"

    if adc_path.exists():
        with open(adc_path) as f:
            adc = json.load(f)

        if "refresh_token" in adc:
            return _refresh_user_token(adc, client_kwargs)

    # 检查服务账号
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path and Path(sa_path).exists():
        with open(sa_path) as f:
            sa = json.load(f)
        return _get_token_from_service_account_dict(sa, client_kwargs)

    raise RuntimeError("未找到凭证，请先运行: gcloud auth application-default login")


def _refresh_user_token(creds: dict, client_kwargs: dict) -> str:
    """用 refresh_token 换 access_token"""
    token_uri = creds.get("token_uri", TOKEN_URI)
    with httpx.Client(**client_kwargs) as client:
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
        return resp.json()["access_token"]


def _get_token_from_service_account_dict(sa: dict, client_kwargs: dict) -> str:
    """从服务账号字典获取 token (需要 cryptography 库)"""
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

    with httpx.Client(**client_kwargs) as client:
        resp = client.post(
            token_uri,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt,
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


def get_token_from_gcloud() -> str:
    """直接从 gcloud CLI 获取 token"""
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


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

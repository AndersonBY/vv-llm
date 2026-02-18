import time
import asyncio
import threading
import importlib
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import cast


def _raise_optional_dependency_error(package_name: str, extra_name: str, exc: ModuleNotFoundError) -> None:
    raise ModuleNotFoundError(f"Optional dependency '{package_name}' is required for this feature. Install with: pip install v-llm[{extra_name}]") from exc


class AsyncRateLimiterBackend(ABC):
    """Rate Limiter Backend Abstract Base Class"""

    @abstractmethod
    async def check_limit(self, key: str, rpm: int, tpm: int, request_cost: int = 1) -> tuple[bool, float]:
        """Returns (allowed, wait_time)"""
        pass


class SyncRateLimiterBackend(ABC):
    """Rate Limiter Backend Abstract Base Class"""

    @abstractmethod
    def check_limit(self, key: str, rpm: int, tpm: int, request_cost: int = 1) -> tuple[bool, float]:
        """Returns (allowed, wait_time)"""
        pass


class AsyncMemoryRateLimiter(AsyncRateLimiterBackend):
    """Async Memory Rate Limiter"""

    def __init__(self):
        self.windows = defaultdict(list)
        self.tokens = defaultdict(int)
        self.lock = asyncio.Lock()

    def _get_last_reset(self, key):
        return self.windows[key][0] if self.windows[key] else time.time()

    async def check_limit(self, key: str, rpm: int, tpm: int, request_cost: int = 1):
        async with self.lock:
            now = time.time()

            # RPM 检查
            window = self.windows[key]
            window = [t for t in window if t > now - 60]
            if len(window) >= rpm:
                return False, 60 - (now - window[0])

            # TPM 检查
            if self.tokens[key] + request_cost > tpm:
                return False, 60 - (now - self._get_last_reset(key))

            window.append(now)
            self.tokens[key] += request_cost
            self.windows[key] = window[-rpm:]
            return True, 0


class SyncMemoryRateLimiter(SyncRateLimiterBackend):
    """Sync Memory Rate Limiter"""

    def __init__(self):
        self.windows = defaultdict(list)
        self.tokens = defaultdict(int)
        self.lock = threading.Lock()

    def _get_last_reset(self, key):
        return self.windows[key][0] if self.windows[key] else time.time()

    def check_limit(self, key: str, rpm: int, tpm: int, request_cost: int = 1) -> tuple[bool, float]:
        """Sync Rate Limiter Check

        Args:
            key: Rate Limiter Key
            rpm: Requests per minute limit
            tpm: Tokens per minute limit
            request_cost: The number of tokens consumed by this request

        Returns:
            Tuple[bool, float]: (allowed, wait_time)
        """
        with self.lock:
            now = time.time()

            # RPM 检查
            window = self.windows[key]
            window = [t for t in window if t > now - 60]
            if len(window) >= rpm:
                return False, 60 - (now - window[0])

            # TPM 检查
            if self.tokens[key] + request_cost > tpm:
                return False, 60 - (now - self._get_last_reset(key))

            window.append(now)
            self.tokens[key] += request_cost
            self.windows[key] = window[-rpm:]
            return True, 0


REDIS_SCRIPT = """
local key = KEYS[1]
local rpm = tonumber(ARGV[1])
local tpm = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])

-- 使用Redis服务器时间（精确到微秒）
local server_time = redis.call('TIME')
local now = tonumber(server_time[1]) + tonumber(server_time[2])/1000000

-- RPM限制检查
local rpm_key = key..'_rpm'
local elements = redis.call('LRANGE', rpm_key, 0, -1)
local valid_elements = {}
local min_valid_time = now - 60

-- 过滤过期时间戳
for _, ts in ipairs(elements) do
    local timestamp = tonumber(ts)
    if timestamp > min_valid_time then
        table.insert(valid_elements, timestamp)
    end
end
local valid_count = #valid_elements

-- 新增：自动清理过期时间戳
if valid_count > 0 then
    redis.call('DEL', rpm_key)
    for _, ts in ipairs(valid_elements) do
        redis.call('RPUSH', rpm_key, ts)
    end
    redis.call('EXPIRE', rpm_key, 60)
end

if valid_count >= rpm then
    local oldest = valid_elements[valid_count]  -- 最旧的有效时间戳
    local remaining = math.max(0.001, 60 - (now - oldest))  -- 保证最小等待时间
    return {0, math.ceil(remaining * 1000)/1000}  -- 保留3位小数
end

-- TPM限制检查（保持不变）
local tpm_key = key..'_tpm'
local current_tokens = tonumber(redis.call('GET', tpm_key) or 0)
if current_tokens + cost > tpm then
    local ttl = redis.call('TTL', tpm_key)
    if ttl < 0 then
        redis.call('SETEX', tpm_key, 60, current_tokens)
        ttl = 60
    end
    return {0, ttl}
end

-- 更新计数（增加时间戳覆盖写入）
redis.call('LPUSH', rpm_key, now)
redis.call('LTRIM', rpm_key, 0, rpm-1)
redis.call('EXPIRE', rpm_key, 60)

redis.call('INCRBY', tpm_key, cost)
redis.call('EXPIRE', tpm_key, 60)

return {1, 0}
"""


class AsyncRedisRateLimiter(AsyncRateLimiterBackend):
    """Async Redis Rate Limiter"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        try:
            import redis.asyncio as redis
        except ModuleNotFoundError as exc:
            _raise_optional_dependency_error("redis", "redis", exc)

        self.redis = redis.Redis(host=host, port=port, db=db)
        self.script = self.redis.register_script(REDIS_SCRIPT)

    async def check_limit(self, key: str, rpm: int, tpm: int, request_cost: int = 1):
        result = await self.script(keys=[key], args=[rpm, tpm, request_cost])
        allowed, wait_time = result
        return bool(allowed), max(1, float(wait_time))


class SyncRedisRateLimiter(SyncRateLimiterBackend):
    """Sync Redis Rate Limiter"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        try:
            import redis
        except ModuleNotFoundError as exc:
            _raise_optional_dependency_error("redis", "redis", exc)

        self.redis = redis.Redis(host=host, port=port, db=db)
        self.script = self.redis.register_script(REDIS_SCRIPT)

    def check_limit(self, key: str, rpm: int, tpm: int, request_cost: int = 1):
        result = self.script(keys=[key], args=[rpm, tpm, request_cost])
        allowed, wait_time = result
        return bool(allowed), max(1, float(wait_time))


class AsyncDiskCacheRateLimiter(AsyncRateLimiterBackend):
    """基于 diskcache 的异步限流器实现"""

    def __init__(self, cache_dir: str = ".rate_limit_cache"):
        """初始化 diskcache 限流器

        Args:
            cache_dir: 缓存目录路径
        """
        try:
            Cache = importlib.import_module("diskcache").Cache
        except ModuleNotFoundError as exc:
            _raise_optional_dependency_error("diskcache", "diskcache", exc)

        self.cache = Cache(cache_dir)
        self._lock = asyncio.Lock()

    def _get_rpm_key(self, key: str) -> str:
        return f"{key}_rpm"

    def _get_tpm_key(self, key: str) -> str:
        return f"{key}_tpm"

    async def check_limit(self, key: str, rpm: int, tpm: int, request_cost: int = 1) -> tuple[bool, float]:
        """检查是否超出限流阈值

        Args:
            key: 限流键
            rpm: 每分钟请求数限制
            tpm: 每分钟令牌数限制
            request_cost: 本次请求消耗的令牌数

        Returns:
            Tuple[bool, float]: (是否允许请求, 需要等待的时间)
        """
        async with self._lock:
            now = time.time()
            rpm_key = self._get_rpm_key(key)
            tpm_key = self._get_tpm_key(key)

            # RPM 检查
            window = cast(list[float], self.cache.get(rpm_key, []) or [])
            window = [t for t in window if t > now - 60]  # 清理过期时间戳

            if len(window) >= rpm:
                return False, 60 - (now - window[0])

            # TPM 检查
            current_tokens = cast(int, self.cache.get(tpm_key, 0))
            if current_tokens + request_cost > tpm:
                # 获取最早的请求时间
                oldest_time = window[0] if window else now
                return False, 60 - (now - oldest_time)

            # 更新状态
            window.append(now)
            window = window[-rpm:]  # 只保留最近的 rpm 个时间戳
            self.cache.set(rpm_key, window, expire=60)
            self.cache.set(tpm_key, current_tokens + request_cost, expire=60)

            return True, 0


class SyncDiskCacheRateLimiter(SyncRateLimiterBackend):
    """基于 diskcache 的同步限流器实现"""

    def __init__(self, cache_dir: str = ".rate_limit_cache"):
        """初始化 diskcache 限流器

        Args:
            cache_dir: 缓存目录路径
        """
        try:
            Cache = importlib.import_module("diskcache").Cache
        except ModuleNotFoundError as exc:
            _raise_optional_dependency_error("diskcache", "diskcache", exc)

        self.cache = Cache(cache_dir)
        self._lock = threading.Lock()

    def _get_rpm_key(self, key: str) -> str:
        return f"{key}_rpm"

    def _get_tpm_key(self, key: str) -> str:
        return f"{key}_tpm"

    def check_limit(self, key: str, rpm: int, tpm: int, request_cost: int = 1) -> tuple[bool, float]:
        """检查是否超出限流阈值

        Args:
            key: 限流键
            rpm: 每分钟请求数限制
            tpm: 每分钟令牌数限制
            request_cost: 本次请求消耗的令牌数

        Returns:
            Tuple[bool, float]: (是否允许请求, 需要等待的时间)
        """
        with self._lock:
            now = time.time()
            rpm_key = self._get_rpm_key(key)
            tpm_key = self._get_tpm_key(key)

            # RPM 检查
            window = cast(list[float], self.cache.get(rpm_key, []) or [])
            window = [t for t in window if t > now - 60]  # 清理过期时间戳

            if len(window) >= rpm:
                return False, 60 - (now - window[0])

            # TPM 检查
            current_tokens = cast(int, self.cache.get(tpm_key, 0))
            if current_tokens + request_cost > tpm:
                # 获取最早的请求时间
                oldest_time = window[0] if window else now
                return False, 60 - (now - oldest_time)

            # 更新状态
            window.append(now)
            window = window[-rpm:]  # 只保留最近的 rpm 个时间戳
            self.cache.set(rpm_key, window, expire=60)
            self.cache.set(tpm_key, current_tokens + request_cost, expire=60)

            return True, 0

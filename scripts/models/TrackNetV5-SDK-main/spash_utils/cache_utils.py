import asyncio
import functools
from typing import Callable
import inspect
from cacheout import Cache


def cached(key: str, ttl: int = 30):
    """
    Cache decorator that locks concurrent calls to the same function with the same key.
    Limitations:
     - Currently not possible to share cache between function calls.
     - Not thread safe
    
    Args:
        key: Format string for the cache key, prefixed with function path.
        ttl: Time to live for the cache
    """
    cache = Cache(ttl=ttl)
    locks = {}
    
    def decorator(func: Callable) -> Callable:
        func.locks = locks
        func.cache = cache
        func.call_count = 0
        func.hit_count = 0
        func.error_count = 0
        func.success_count = 0
        func.cache_clear = lambda: cache.clear()

        @functools.wraps(func) 
        async def wrapper(*args, **kwargs):
            # Format the key with args/kwargs
            # Get function signature and bind args/kwargs
            locks = wrapper.locks
            cache = wrapper.cache
    
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Format key using bound arguments
            formatted_key = f"{key.format(**bound_args.arguments)}"

            # Get or create lock for this key
            if formatted_key not in locks:
                locks[formatted_key] = asyncio.Lock()
            lock = locks[formatted_key]

            async with lock:
                # Check cache first using proper Cache API
                if cache.has(formatted_key):
                    wrapper.hit_count += 1
                    return cache.get(formatted_key)

                # Call function and cache result
                try:
                    result = await func(*args, **kwargs)
                    wrapper.success_count += 1
                except Exception as e:
                    wrapper.error_count += 1
                    raise e
                finally:
                    wrapper.call_count += 1
                cache.set(formatted_key, result)
                return result
        
        return wrapper
    return decorator

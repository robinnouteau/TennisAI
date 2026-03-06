import traceback
from loguru import logger


async def wrap_task(func, *args, fields: dict = None, **kwargs) -> dict:
    """Wraping a task (async function) call to return a dict with fields,
    result, exception and traceback.

    This wraper is useful to create a iterable of awaiable tasks for any number of
    center/services and being able to await all of them at once using asyncio.as_completed.

    Returns:
        dict: A dictionary with fields, result, exception and traceback.
    """
    
    o = {
        **fields,
        'result': None,
        'exc': None,
        'trace': None,
    }
    try:
        r = await func(*args, **kwargs)
        if isinstance(r, tuple) and len(r) == 2:
            response, ctxt = r
        else:
            raise ValueError(f"Invalid return type {type(r)}={r}, expected tuple of length 2")
        
        o["result"] = response
        o.update(ctxt)
    except Exception as e:
        fields_str = ", ".join([f"{k}: {v}" for k, v in fields.items()])
        logger.error(f"Error wrapping task {fields_str} {func} {args} {kwargs} exc:{type(e)} {traceback.format_exc()}")
        o.update(ctxt)
        o["exc"] = e
        o["trace"] = traceback.format_exc()
    return o
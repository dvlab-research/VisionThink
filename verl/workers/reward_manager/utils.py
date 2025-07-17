import signal
import ray
from typing import Callable, Any

# Keep this outside the main wrapper function for clarity and efficiency.
def _timeout_handler(signum, frame):
    """Signal handler function to raise a TimeoutError."""
    # print("Signal handler called!") # Debugging
    raise TimeoutError("Operation timed out!")


@ray.remote
def reward_func_timeout_ray(func: Callable, timeout_seconds: int, *args: Any, **kwargs: Any):
    """A decorator that applies a timeout to the decorated function using signal.

    Args:
        timeout_seconds (int): Number of seconds before timing out the decorated function.
            Defaults to 10 seconds.

    Notes:
        Only works on Unix systems as it uses signal.alarm.
    """                
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return func(*args, **kwargs)
    except TimeoutError:
        return {"score": 0.0, "extra_info": {"is_filter": "1"}}
    finally:
        # cancel alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

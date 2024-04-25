from __future__ import annotations

from functools import reduce
from threading import Thread
from time import time, sleep
from traceback import print_exc
from typing import Callable, Type, Any, Iterable
from json import load


def convert_to_dataclass(cls: Type, d: dict) -> Any:
    """

    Args:
        cls: Class to convert the dictionary to.
        d: Dictionary containing the dataclass data.

    Returns:
        Instance of cls containing the given data in d.

    """
    return cls(**d)


def read_json(path: str) -> dict | list:
    """

    Args:
        path: Path to the JSON file.

    Returns:
        JSON data in the form of a dict or list, depending on the JSON data.

    """
    with open(path, 'r') as file:
        return load(file)


def make_threaded(func: Callable) -> Callable[..., Thread]:
    """
    Wraps the every function in a thread.

    Args:
        func: Function to wrap.

    Returns:
        Thread wrapping the function.

    """

    return lambda *args, **kwargs: Thread(target=func, args=args, kwargs=kwargs)


@make_threaded
def every(delay: float | int, task: Callable, *args: Any, **kwargs: Any) -> None:
    """

    Args:
        delay: Delay for the job in seconds.
        task: Function to be executed.
        *args: Arguments given to the task function.
        **kwargs: Keyword arguments given to the task function.

    """

    next_time: float = time() + delay

    while True:
        sleep(
            max(
                0.0,
                next_time - time()
            )
        )

        try:
            task(*args, **kwargs)
        except Exception:
            print_exc()
            exit(1)

        # skip tasks if we are behind schedule:
        next_time += (time() - next_time) // delay * delay + delay


def join_jsons(jss: Iterable[str]) -> str:
    """

    Args:
        jss: Iterable of JSONs in the form of strings.

    Returns:
        The combined strings into a single valid JSON.

    """
    return reduce(lambda j0, j1: f"{{{j0[1:-1]}, {j1[1:-1]}}}", jss)


def format_sse(data: str) -> str:
    """

    Args:
        data: Data to be sent over SSE.

    Returns:
        A formatted string valid for SSE.

    """
    return f"data: {data}\n\n"

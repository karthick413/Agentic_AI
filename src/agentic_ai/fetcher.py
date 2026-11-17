import asyncio
import random
import time
from dataclasses import dataclass

import httpx

# HTTP status codes that indicate temporary problems.
# These errors may disappear if we try the request again.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# dataclass automatically creates methods like __init__ for this class.
# This class stores the result of each URL fetch operation.
@dataclass
class FetchResult:
    url: str  # URL that was requested
    ok: bool  # True if request succeeded, otherwise False
    status: int | None = None  # HTTP status code, or None if no response was received
    attempts: int = 0  # Total number of attempts made
    elapsed: float = 0.0  # Total time taken, in seconds
    size: int = 0  # Response body size in bytes
    error: str | None = None  # Error message, if the request failed


# Asynchronously fetch one URL, retrying when temporary errors occur.
async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    timeout: float = 5.0,
) -> FetchResult:

    # Record the starting time so we can calculate total elapsed time.
    start = time.perf_counter()

    # Store the latest HTTP status code and error message.
    status: int | None = None
    last_error: str | None = None

    # Make one initial attempt plus the allowed number of retries.
    # Example: retries=3 means attempts 1, 2, 3, and 4.
    for attempt in range(1, retries + 2):
        try:
            # Set a maximum time limit for this HTTP request.
            # If the request takes too long, asyncio raises TimeoutError.
            async with asyncio.timeout(timeout):
                # Send the HTTP GET request asynchronously.
                # await allows other async tasks to run while waiting.
                resp = await client.get(url)

            # Extract the HTTP status code from the response.
            status = resp.status_code

            # If the status is not retryable, return the result immediately.
            # This includes successful responses and errors such as HTTP 404.
            if status not in RETRYABLE_STATUS:
                return FetchResult(
                    url=url,
                    ok=resp.is_success,  # True for successful HTTP status codes
                    status=status,
                    attempts=attempt,
                    elapsed=time.perf_counter() - start,
                    size=len(resp.content),  # Number of response bytes
                    error=None if resp.is_success else f"HTTP {status}",
                )

            # If the status is retryable, save the error and continue.
            last_error = f"HTTP {status}"

        # Catch network/connection errors and request timeouts.
        # These errors are treated as temporary failures worth retrying.
        except (httpx.TransportError, TimeoutError) as exc:
            # Save the exception type, such as ConnectError or TimeoutError.
            last_error = type(exc).__name__

        # If another attempt is allowed, wait before retrying.
        if attempt <= retries:
            # Exponential backoff:
            # Each retry waits longer than the previous one.
            # Example with base_delay=0.5:
            # First retry: 0.5 seconds
            # Second retry: 1.0 seconds
            # Third retry: 2.0 seconds
            delay = base_delay * 2 ** (attempt - 1)

            # Add a small random delay (jitter).
            # This helps prevent many clients from retrying simultaneously.
            delay += random.uniform(0, delay * 0.1)

            # Wait asynchronously without blocking the event loop.
            await asyncio.sleep(delay)

    # All attempts have failed.
    # Return a failure result containing the final error information.
    return FetchResult(
        url=url,
        ok=False,
        status=status,
        attempts=retries + 1,
        elapsed=time.perf_counter() - start,
        error=last_error,
    )


# Fetch multiple URLs concurrently while limiting the number
# of requests that can run at the same time.
async def fetch_all(
    urls: list[str],
    *,
    max_concurrency: int = 5,
    transport: httpx.AsyncBaseTransport | None = None,
    **kwargs,
) -> list[FetchResult]:

    # Semaphore controls how many requests can run simultaneously.
    # For example, Semaphore(5) allows at most 5 requests at a time.
    sem = asyncio.Semaphore(max_concurrency)

    # Create one shared asynchronous HTTP client for all URLs.
    # follow_redirects=True automatically follows HTTP redirects.
    # transport can be supplied for testing with MockTransport.
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        # Inner function that fetches one URL while respecting the semaphore.
        async def bounded(url: str) -> FetchResult:

            # Wait here if the concurrency limit has already been reached.
            async with sem:
                # Fetch the URL with retry support.
                # **kwargs forwards additional options such as retries or timeout.
                return await fetch_with_retry(client, url, **kwargs)

        # Start fetching all URLs concurrently.
        # gather waits for all results and returns them in input order.
        return await asyncio.gather(*(bounded(u) for u in urls))

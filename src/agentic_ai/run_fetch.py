import asyncio
import time

# Import fetch_all from our fetcher.py module.
# This function fetches multiple URLs concurrently.
from agentic_ai.fetcher import fetch_all

# Create a list of 20 URLs.
# The first 15 URLs simulate slow requests (approximately 1-second delay).
URLS = ["https://httpbin.org/delay/1"] * 15 + [
    "https://httpbin.org/status/404",  # Returns HTTP 404 (Not Found)
    "https://httpbin.org/status/503",  # Returns HTTP 503 (Service Unavailable)
    "https://example.com",  # A real website
    "https://httpbin.org/status/200",  # Returns HTTP 200 (Success)
    "https://httpbin.org/get",  # Returns information about the GET request
]


# Define the main asynchronous function.
async def main():

    # Run the same experiment with three different concurrency limits:
    # 1 = One request at a time
    # 5 = Up to five requests simultaneously
    # 20 = Up to twenty requests simultaneously
    for limit in (1, 5, 20):
        # Record the starting time before fetching the URLs.
        # perf_counter() provides a high-resolution timer.
        t = time.perf_counter()

        # Fetch all 20 URLs using the selected concurrency limit.
        results = await fetch_all(
            URLS,
            max_concurrency=limit,  # Maximum simultaneous requests
            retries=2,  # Retry temporary failures up to 2 times
            base_delay=0.3,  # Initial delay between retries (seconds)
        )

        # Count how many requests were successful.
        # r.ok is True for success and False for failure.
        # Python treats True as 1 and False as 0.
        ok = sum(r.ok for r in results)

        # Display the experiment results:
        # limit = concurrency setting
        # ok = number of successful requests
        # len(results) = total number of results
        # perf_counter() - t = total execution time
        print(
            f"concurrency={limit:>2}  "
            f"ok={ok}/{len(results)}  "
            f"time={time.perf_counter() - t:.1f}s"
        )


# Start the asynchronous program.
# asyncio.run() creates an event loop and executes main().
asyncio.run(main())

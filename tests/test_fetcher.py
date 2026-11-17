import asyncio

import httpx

# Import the two functions we want to test from fetcher.py
from agentic_ai.fetcher import fetch_all, fetch_with_retry

# A sample URL used by all tests.
# MockTransport will intercept requests, so no real website is contacted.
URL = "https://example.test/a"


# TEST 1: Check whether a successful request works correctly
async def test_success():

    # Create a fake HTTP transport that always returns:
    # HTTP 200 = successful response
    # Response body = "hello"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text="hello"))

    # Create an asynchronous HTTP client using the fake transport
    async with httpx.AsyncClient(transport=transport) as client:
        # Call the function being tested and wait for its result
        r = await fetch_with_retry(client, URL)

    # Check that:
    # 1. The request was successful
    # 2. It took only one attempt
    # 3. The response body contains 5 bytes ("hello")
    assert r.ok and r.attempts == 1 and r.size == 5


# TEST 2: Check whether failed requests are retried and eventually succeed
async def test_retries_then_succeeds():

    # Dictionary used to count how many times the fake server is called
    calls = {"n": 0}

    def handler(request):

        # Increase the request counter every time a request arrives
        calls["n"] += 1

        # Return HTTP 503 for the first two calls.
        # Return HTTP 200 on the third call.
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, text="ok")

    # Create an asynchronous client with the fake transport
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # Allow up to 3 retries, with a short delay between attempts
        r = await fetch_with_retry(client, URL, retries=3, base_delay=0.01)

    # Check that the request eventually succeeded on attempt number 3
    assert r.ok and r.attempts == 3


# TEST 3: Check whether the function stops after exhausting its retries
async def test_gives_up_after_retries():

    # Fake server always returns HTTP 500 (server error)
    transport = httpx.MockTransport(lambda r: httpx.Response(500))

    async with httpx.AsyncClient(transport=transport) as client:
        # retries=2 means:
        # 1 initial attempt + 2 retries = 3 total attempts
        r = await fetch_with_retry(client, URL, retries=2, base_delay=0.01)

    # Check that:
    # 1. The request failed
    # 2. There were 3 total attempts
    # 3. The error message is "HTTP 500"
    assert not r.ok and r.attempts == 3 and r.error == "HTTP 500"


# TEST 4: Check whether HTTP 404 is handled without retrying
async def test_404_not_retried():

    # Fake server always returns HTTP 404 (not found)
    transport = httpx.MockTransport(lambda r: httpx.Response(404))

    async with httpx.AsyncClient(transport=transport) as client:
        # Even though 3 retries are allowed,
        # HTTP 404 is not a retryable status in fetcher.py
        r = await fetch_with_retry(client, URL, retries=3, base_delay=0.01)

    # Check that the request failed but was attempted only once
    assert not r.ok and r.attempts == 1


# TEST 5: Check whether connection errors are retried
async def test_connection_error_is_retried():

    def handler(request):

        # Simulate a connection failure instead of returning an HTTP response
        raise httpx.ConnectError("boom")

    # Create an asynchronous client with the fake transport
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # Allow 2 retries after the initial failed attempt
        r = await fetch_with_retry(client, URL, retries=2, base_delay=0.01)

    # Check that:
    # 1. The request ultimately failed
    # 2. There were 3 total attempts
    # 3. The recorded error type is "ConnectError"
    assert not r.ok and r.attempts == 3 and r.error == "ConnectError"


# TEST 6: Check whether the concurrency limit is respected
async def test_concurrency_limit_respected():

    # active = number of requests currently running
    # peak = highest number of requests running at the same time
    active = peak = 0

    async def handler(request):

        # Use nonlocal because we want to modify variables
        # defined in the outer test function
        nonlocal active, peak

        # A new request has started, so increase active count
        active += 1

        # Record the highest number of simultaneous requests
        peak = max(peak, active)

        # Simulate network waiting for 0.05 seconds
        await asyncio.sleep(0.05)

        # The request has finished, so decrease active count
        active -= 1

        # Return a successful HTTP response
        return httpx.Response(200, text="x")

    # Create 20 different URLs to fetch
    urls = [f"https://example.test/{i}" for i in range(20)]

    # Fetch all 20 URLs while allowing a maximum of 5 concurrent requests
    # MockTransport simulates the responses without using the real network
    results = await fetch_all(
        urls, max_concurrency=5, transport=httpx.MockTransport(handler)
    )

    # Check that:
    # 1. Every request succeeded
    # 2. More than one request ran concurrently
    # 3. No more than 5 requests ran at the same time
    assert all(r.ok for r in results)
    assert 1 < peak <= 5

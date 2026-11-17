import asyncio
import time

# ============================================================
# EXPERIMENT 1: SEQUENTIAL VS GATHER
# ============================================================


async def work(name, delay):
    # asyncio.sleep() pauses this coroutine without blocking
    # the event loop. Other tasks can run while it waits.
    await asyncio.sleep(delay)
    return name


async def experiment_1():

    print("\n===== EXPERIMENT 1 =====")

    # Sequential: each task waits for the previous task.
    # Three tasks x 1 second = approximately 3 seconds.

    start = time.perf_counter()

    for i in range(3):
        await work(i, 1)

    print("Sequential:", round(time.perf_counter() - start, 1), "seconds")

    # Concurrent: gather() coordinates all three operations.
    # They can wait at the same time.
    # Expected duration = approximately 1 second.

    start = time.perf_counter()

    await asyncio.gather(*(work(i, 1) for i in range(3)))

    print("Gather:", round(time.perf_counter() - start, 1), "seconds")


# ============================================================
# EXPERIMENT 2: SEMAPHORE
# ============================================================

# Semaphore(2) allows a maximum of two tasks to enter
# the protected section at the same time.

sem = asyncio.Semaphore(2)


async def limited(i):

    async with sem:
        print("Start", i)

        # Simulate an API request taking one second.
        await asyncio.sleep(1)

        print("End", i)


async def experiment_2():

    print("\n===== EXPERIMENT 2 =====")

    # Launch six tasks concurrently.
    await asyncio.gather(*(limited(i) for i in range(6)))

    # Expected: only two tasks run at a time.
    # Six tasks / two at a time = three batches.
    # Total duration approximately 3 seconds.


# ============================================================
# EXPERIMENT 3: TIMEOUT
# ============================================================


async def experiment_3():

    print("\n===== EXPERIMENT 3 =====")

    try:
        # Allow a maximum of one second.
        async with asyncio.timeout(1):
            # This operation wants to wait five seconds.
            await asyncio.sleep(5)

    except TimeoutError:
        # Timeout cancels the operation and raises TimeoutError.
        print("Timed out!")

    # Expected duration: approximately one second.


# ============================================================
# EXPERIMENT 4: BLOCKING MISTAKE
# ============================================================


async def blocking_work(name, delay):

    # WRONG: time.sleep() blocks the event-loop thread.
    # Other asynchronous tasks cannot progress during this wait.
    time.sleep(delay)

    return name


async def experiment_4():

    print("\n===== EXPERIMENT 4 =====")

    start = time.perf_counter()

    # Although gather() is used, time.sleep() blocks
    # the event loop each time a task runs.
    await asyncio.gather(*(blocking_work(i, 1) for i in range(3)))

    print("Blocking gather:", round(time.perf_counter() - start, 1), "seconds")

    # Expected duration: approximately 3 seconds.
    # The concurrency advantage disappears.


# ============================================================
# EXPERIMENT 5: GATHER AND EXCEPTIONS
# ============================================================


async def success():

    await asyncio.sleep(1)
    return "Success"


async def failure():

    await asyncio.sleep(0.5)

    # Deliberately raise an exception.
    raise ValueError("Something went wrong!")


# CASE A: Without return_exceptions=True


async def gather_without_exceptions():

    print("\n--- Without return_exceptions ---")

    try:
        results = await asyncio.gather(success(), failure())

        print(results)

    except ValueError as error:
        # By default, gather propagates the exception.
        print("Caught error:", error)

    # The other submitted task is not automatically cancelled.
    # It may continue running after the exception is propagated.


# CASE B: With return_exceptions=True


async def gather_with_exceptions():

    print("\n--- With return_exceptions=True ---")

    results = await asyncio.gather(success(), failure(), return_exceptions=True)

    # Exceptions are returned as items in the results list.
    for result in results:
        if isinstance(result, Exception):
            print("Error:", result)

        else:
            print("Result:", result)


# ============================================================
# MAIN FUNCTION
# ============================================================


async def main():

    await experiment_1()

    await experiment_2()

    await experiment_3()

    await experiment_4()

    await gather_without_exceptions()

    await gather_with_exceptions()

    print("\n===== ALL EXPERIMENTS COMPLETED =====")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

# asyncio.run() starts the event loop and runs main().
asyncio.run(main())

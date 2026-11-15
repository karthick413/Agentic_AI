def add(a, b) -> int:
    return a + b


print(add(2, 3))


def is_even(n) -> bool:
    return n % 2 == 0


print(is_even(4))


def title_cases_trim(s) -> str:
    return s.title().strip()


print(title_cases_trim("hello world"))


def avg(numbers) -> float:
    return sum(numbers) / len(numbers) if numbers else 0


print(avg([1, 2, 3, 4, 5]))
print(avg([]))


def repeat_string(s, n) -> str:
    return s * n


print(repeat_string("abc", 3))

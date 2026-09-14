import re

def isPalindrome(text: str) -> bool:
    return text == text[::-1]

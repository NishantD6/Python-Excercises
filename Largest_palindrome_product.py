#!/usr/bin/env python3

"""
A palindromic number reads the same both ways. 
The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.

Find the largest palindrome made from the product of two 3-digit numbers.
"""

n = 999
list_of_palindromes = []
for x in range(1, n):
	for y in range(1, n):
		palindromes = x*y
		if str(palindromes) == str(palindromes)[::-1]:
			list_of_palindromes.append(palindromes)
a = max(list_of_palindromes)
print(a)



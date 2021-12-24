#!/usr/bin/env python3
"""
The sum of the squares of the first ten natural numbers is 385
The square of the sum of the first ten natural numbers is 3025
Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is 2640. 

Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.

"""
max_num = 100

sum_of_sq = 0
sq_of_sum = 0

for i in range(1, max_num+1):
	sum_of_sq = sum_of_sq + i*i
	sq_of_sum = sq_of_sum + i
print(sq_of_sum*sq_of_sum -sum_of_sq)
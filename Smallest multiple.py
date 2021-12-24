#!/usr/bin/env python3

"""
2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.

What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?

"""
num_list = []
max_num = 20
for i in range(1, max_num+1):
	number = i
	for m in num_list:
		if number%m == 0:
			number = number/m
		else:
			pass
	num_list.append(int(number))
	
result =1 
for i in num_list:
	result = result*i
print(result)
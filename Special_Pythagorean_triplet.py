#!/usr/bin/env python3

# A Pythagorean triplet is a set of three natural numbers, a < b < c, for which, a2 + b2 = c2
# For example, 32 + 42 = 9 + 16 = 25 = 52.

# There exists exactly one Pythagorean triplet for which a + b + c = 1000.
# Find the product abc.

max_number = 500
target = 1000
for i in range(max_number+1, 0, -1):
	for m in range(1, max_number+1):
		if i <= m:
			pass
		elif ((i*i - m*m)**0.5).is_integer() == True:
			if i+m +int((i*i - m*m)**0.5) == target:
				print(i*m* int((i*i - m*m)**0.5))     
"""
Letter counter
1- Make words small letters
2- Set up a counter to count all letters in the word/statement

"""

def lettercounter(a):
    a=a.lower()
    counter = {}
    for letter in a:
        if letter in counter:
            counter[letter]+=1
        else:
            counter[letter]=1
    return counter
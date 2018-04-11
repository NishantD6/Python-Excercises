"""
Anagram Checker
1st - Make all letters small and remove spaces
2nd - First check if the count of words in A == B
3rd - Create a counter to count number of words in A
4th - Remove count of words in B
5th - Check if they are the same

"""
#Make All letters small and remove space
def anagram(a, b):
      a =  a.replace(" ","").lower()
      b = b.replace(" ", "").lower()

#First check if the count of words in A == B

      if (len(a)!=len(b)):
             return False
#Create a counter to count number of words in A
      counter = {}
      for letter in a:
             if letter in counter:
                    counter[letter]+=1
             else:
                    counter[letter] = 1

#Remove count of words in B
       for letter in b:
              if letter in counter:
                     counter[letter]-=1
              else:
                     return False

       for k in counter:
              if counter[k]!=0:
                     return False
    else:
              return True
       pass
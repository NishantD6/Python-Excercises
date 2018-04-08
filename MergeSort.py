def rank(alist):
       n= len(alist)
       if n>1:
              if n%2==0:
                     half=int(n/2)
                     n1=alist[0:half]
                     n2=alist[half:n]

              else:
                     half=int((n+1)/2)
                     n1 = alist[0:half]
                     n2 = alist[half:n]

              rank(n1)
              rank(n2)

              i=0
              j=0
              k=0

              while i<len(n1) and j<len(n2):
                     if n1[i]<n2[j]:
                            alist[k]=n1[i]
                            i=i+1
                     else:
                            alist[k]=n2[j]
                            j=j+1
                      k=k+1
              while  i< len(n1):
                     alist[k]=n1[i]
                     i=i+1
                     k=k+1
              while j < len(n2):
                     alist[k] = n2[j]
                     j = j + 1
                     k = k + 1
       print(alist)
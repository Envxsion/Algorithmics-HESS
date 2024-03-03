'''You will write functions to convert between decimal and binary representation in Python.'''
def dec2bin(n):
    #return bin(n).replace("0b", "") is a oneliner that could be used instead
    k=[]
    while (n>0):
        k.append(int(float(n%2)))
        n=(n-int(float(n%2)))/2
    k.reverse() 
    string = ''.join(str(k) for k in k)
    return string

def bin2dec(m):
    #return the int(m, 2) oneliner works too
    dec= 0
    for digit in m:
        dec= dec * 2 + int(digit)
    return dec


print(dec2bin(int(input("Enter a number to conv to binary: "))))
print(bin2dec(input("Enter a number to conv to decimal: ")))
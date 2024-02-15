import math
'''Write a function which finds the largest power of 2 which goes into a number. Do this using a loop rather than logarithms. Remember that we use ** to raise a number to a power in Python
'''

def largest_power_of_2(num):
    i = 0
    while 2**i <= num:
        i += 1
    return i-1

print(largest_power_of_2(int(input("Enter a number: "))))

'''Next write a function which finds the number of digits in a number. You can use some of the functions listed below (e.g. logarithms), and there are multiple possible approaches.'''

def digits(num2):
    return len(str(num2))
print(digits(int(input("Enter a number: "))))
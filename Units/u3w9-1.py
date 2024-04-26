'''
Recursive Functions
Write recursive functions in Python for each of the following. 
1.	Calculate the factorial of a number.
2.	Reverse a string letter by letter.
3.	Find the nth term in the Fibonacci sequence.
4.	(Harder) generate all possible permutations of a list of unique elements.
Make sure you include a base case. As always, include pseudocode-like comments
'''

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def reverse_string(string):
    def swap_chars(s, i, j):
        if i < j:
            s[i], s[j] = s[j], s[i]
            swap_chars(s, i + 1, j - 1)

    str_list = list(string)
    swap_chars(str_list, 0, len(str_list) - 1)
    return ''.join(str_list)

print(reverse_string("hello")) 

def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def permutations(elements):
    if len(elements) == 1:
        return [elements]
    else:
        all_perms = []
        for i in range(len(elements)):
            perms_without_i = permutations(elements[:i] + elements[i+1:])
            for perm in perms_without_i:
                all_perms.append([elements[i]] + perm)
        return all_perms

print("Factorial :", factorial(6))
print("Reverse: ", reverse_string("hi how do u do?"))
print("Fibonacci sequence up to the 6th term: ", [fibonacci(i) for i in range(6)])
print("Permutations: ", permutations([1, 2, 3,4]))

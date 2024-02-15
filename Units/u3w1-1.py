import random
import math
# This code creates two random numbers between 1 and 50
first = random.randint(1,50)
second = random.randint(1,50)

# ADD YOUR CODE in the relevant spaces below.

print(f"My random numbers are: {first} and {second}")

# 2. Store the numbers in variables 'larger' and 'smaller'. State the larger number.
# (if they are the same then larger = smaller.)

smaller = min(first, second)
larger = max(first, second)

# 3. State which of your numbers are even.
even = print("1") if (first % 2 == 0) else print("2") if (second % 2 == 0) else print("Both") if (first % 2 == 0) and (second % 2 == 0) else print("None")
# 4. State whether or not the larger number is exactly divisible by the smaller
#    number. If not, state the remainder.
div = print("True") if larger % smaller else print(larger % smaller)
# 5. If the two sides are the hypotenuse and shorter side of a right angled triangle,
#    find the length of the other shorter side.
short_side = print(math.sqrt(larger**2 - smaller**2))
# Run your code multiple times to make sure it works as expected.

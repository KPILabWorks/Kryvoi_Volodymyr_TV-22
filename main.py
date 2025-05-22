from functools import reduce

numbers = [10, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
print("Добуток: ", product)

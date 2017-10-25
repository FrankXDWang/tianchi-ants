from functools import reduce
num = input('enter a number:')
s = str(num)
def f(s):
    dic = {'1':1,'2':2}
    return dic[s]
def add(x,y):
    return x+y
s = map(f, s)
print(num,list(s))
b = list(s)
print(b)
_sum = reduce(add,[1,2])
print(_sum)

#ok, cool. I didn't write py for 8 mon.



class test:
    invisible:int
    def __init__(self):
        self.invisible = 111
        temp_1_2 = test.____step1((333,444))
        test.step2(temp_1_2)
        pass
    @staticmethod
    def ____step1(context):
        one_num, other_num = context
        third_number = one_num + other_num
        return one_num, other_num, third_number
    @staticmethod
    def step2(context):
        one_num, other_num, third_number = context
        print(one_num)
        print(other_num)
        print(third_number)
        pass
    pass
t = test()
test.____step1()

assert False        












for i in range(6,3,-1):
    print(i)
assert False

print(not False and not False)

a = int( "1000", 2)
b = 0b111
s = "{:b}".format(a)
ss = s[3]


class A:
    b:'B'
    pass
class B:
    a:A
    pass
#ok, cool. I didn't write py for 8 mon.



class test:
    def __init__(self):
        self.FieldLength = 4
        self.bitmask = 0b1100
        self.addr = 0b0100
        self.dataset = [(0b0101,True)]
        pass
    def detect_best_bit_to_split(self):
        temp:list[float]
        # str_bitmask = "{:b}".format(self.bitmask)
        # str_bitmask_len = str_bitmask.__len__()
        # start = str_bitmask_len-self.FieldLength
        for i in range(self.FieldLength-1,-1,-1):
            # bit = 0
            # if i>=0:
            #     bit = int(str_bitmask[i], 2)
            #     pass
            # if bit=0
            one_shift_by_i:int = 1<<i
            bit_of_bitmask_for_this_i = one_shift_by_i&self.bitmask
            if bit_of_bitmask_for_this_i != 0:
                #if this bit is not marked in bitmask. Otherwise ignore this i.
                continue
            total = 0
            same = 0
            bit_of_addr_for_this_i = self.addr&one_shift_by_i
            for item in self.dataset:
                total = total+1
                bit_of_addr_of_item = item[0] & one_shift_by_i
                if bit_of_addr_for_this_i == bit_of_addr_of_item:
                    same = same+1
a = test()
a.detect_best_bit_to_split()
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
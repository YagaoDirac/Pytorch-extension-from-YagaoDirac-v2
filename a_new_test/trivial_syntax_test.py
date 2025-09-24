#ok, cool. I didn't write py for 8 mon.



class test:
    def __init__(self):
        self.FieldLength = 5
        self.bitmask = 0b11000
        #self.addr = 0b01000
        self.dataset = [
            (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,True),
            (0b01100,True),(0b01101,True),(0b01110,True),(0b01111,True),
                        ]
        pass
    def detect_best_bit_to_split(self):
        actual_index:list[int] = []
        num_of_same:list[int] = []
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
                #This bit is in bitmask, ignore this i.
                continue
            #total = 0
            dot_product_like = 0
            #Y is same as this bit. If a and result is true, +1. If a' and result is false, +1. Otherwise +0.
            #bit_of_addr_for_this_i = self.addr&one_shift_by_i#always 0????
            for item in self.dataset:
                #total = total+1
                bit_of_addr_of_item_raw = item[0] & one_shift_by_i
                bit_of_addr_of_item = bit_of_addr_of_item_raw!=0
                #if not (bit_of_addr_of_item ^ item[1]):
                logic_var_xor_this_result:bool = bit_of_addr_of_item ^ item[1]
                same_minus_this = 1-2*int(logic_var_xor_this_result)
                dot_product_like = dot_product_like + same_minus_this
                pass
            actual_index.append(i)
            num_of_same.append(dot_product_like)
            pass#for i in range
        
        print(actual_index)
        print(num_of_same)
        pass#end of function.
    
a = test()
a.FieldLength = 5
a.bitmask = 0b11000
a.dataset = [
    (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,True),
    (0b01100,True),(0b01101,True),(0b01110,True),(0b01111,True),
                ]
a.detect_best_bit_to_split()
print()

a = test()
a.FieldLength = 5
a.bitmask = 0b11001
a.dataset = [
    (0b01001,True),(0b01011,True),
    (0b01101,True),(0b01111,True),
                ]
a.detect_best_bit_to_split()
print()

a = test()
a.FieldLength = 5
a.bitmask = 0b11000
a.dataset = [
    (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,True),
    #(0b01100,True),(0b01101,True),(0b01110,True),(0b01111,True),
                ]
a.detect_best_bit_to_split()
print()



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
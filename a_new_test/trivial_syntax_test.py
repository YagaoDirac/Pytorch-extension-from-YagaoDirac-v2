#ok, cool. I didn't write py for 8 mon.



class test:
    FieldLength:int
    bitmask:int
    addr:int
    dataset:list[tuple[int,bool]]
    def __init__(self):
        # self.FieldLength = 5
        # self.bitmask = 0b11000
        # self.addr = 0b01000
        # self.dataset = []
        pass
    def _check_all_addr(self)->None:
        reversed_bitmask = ~self.bitmask
        addr_bits_out_of_bitmask = self.addr&reversed_bitmask
        assert 0 == addr_bits_out_of_bitmask, "self.addr is bad. It has bit set to 1 outside the bitmask."
        masked_addr = self.addr&self.bitmask
        for item in self.dataset:
            masked_addr_of_item = item[0]&self.bitmask
            assert masked_addr == masked_addr_of_item
            pass
        pass
    def detect_best_bit_to_split(self)->None:
        actual_index:list[int] = []
        num_of_same:list[int] = []
        for i in range(self.FieldLength-1,-1,-1):
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

if "check_all_addr" and False:
    a = test()
    a.bitmask = 0b11000
    a.addr =    0b01000
    a.dataset = [
        (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,True),
        (0b01100,True),(0b01101,True),(0b01110,True),(0b01111,True),
                    ]
    a.check_all_addr()
    pass



a = test()
a.FieldLength = 5
a.bitmask = 0b11000
a.addr = 0b01000
a.dataset = [
    (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,False),
    (0b01100,True),(0b01101,True),(0b01110,False),#(0b01111,False),
                ]
a._check_all_addr()
a.detect_best_bit_to_split()
print()

a = test()
a.FieldLength = 5
a.bitmask = 0b11001
a.dataset = [
    (0b01001,True),(0b01011,True),
    (0b01101,True),(0b01111,True),
                ]
a._check_all_addr()
a.detect_best_bit_to_split()
print()

a = test()
a.FieldLength = 5
a.bitmask = 0b11000
a.dataset = [
    (0b01000,True),(0b01001,True),(0b01010,True),(0b01011,True),
    #(0b01100,True),(0b01101,True),(0b01110,True),(0b01111,True),
                ]
a._check_all_addr()
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
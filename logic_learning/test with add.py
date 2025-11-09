from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

import numpy

from logic_learning_v1 import *



if "a special case" and True:
    #011:10, 111:11
    bits_count = 1
    training_Dataset_Set = Dataset_Set(bits_count*2+1, bits_count+1)
    addr = 0b011
    training_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    addr = 0b111
    training_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    training_Dataset_Set.sort()
    assert training_Dataset_Set.max_input_bits == 3
    assert training_Dataset_Set.get_recommended_addr_FieldLength() == 3
    assert training_Dataset_Set.get_output_count() == 2
    assert training_Dataset_Set.get_sample_count___maybe_unsafe() == 2
    
    
    readable_dataset = training_Dataset_Set.get_readable___check_btw()
    assert readable_dataset == "011:10, 111:11"
    a_DatasetField_Set = DatasetField_Set(training_Dataset_Set)
    assert a_DatasetField_Set.get_input_count() == 3
    assert a_DatasetField_Set.get_output_count() == 2
    finished = a_DatasetField_Set.valid(training_Dataset_Set)[0]
    assert finished

    #a bit manual check.        
    _tree_0_the_high_bit = a_DatasetField_Set.fields[0].readable_as_tree()
    assert _tree_0_the_high_bit == "___:1+ir"
    _tree_1_the_low_bit = a_DatasetField_Set.fields[1].readable_as_tree()
    assert _tree_1_the_low_bit == "___:(0__:0+ir, 1__:1+ir)"
    
    
    #now the test set. It's bigger than the training set.        
    #000, 011, 100, 111
    test_Dataset_Set = Dataset_Set(bits_count*2+1, bits_count+1)
    addr = 0b000
    test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    addr = 0b011
    test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    addr = 0b100
    test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    addr = 0b111
    test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    test_Dataset_Set.sort()
    
    assert test_Dataset_Set.max_input_bits == 3
    assert test_Dataset_Set.get_recommended_addr_FieldLength() == 3
    assert test_Dataset_Set.get_output_count() == 2
    assert test_Dataset_Set.get_sample_count___maybe_unsafe() == 4
    
    readable_dataset = test_Dataset_Set.get_readable___check_btw()
    assert readable_dataset == "000:00, 011:10, 100:01, 111:11"
    _result_tuple_bbsil = a_DatasetField_Set.valid(test_Dataset_Set)
    finished = _result_tuple_bbsil[0]
    assert finished
    '''in this case, one bit is random, but the other bit passes all the 4 cases.'''
    
    pass

if "temporal code. gonna get into a standard process." and True:
    #input zone.
    input_bit_count = 1
    
    training_Dataset_Set = Dataset_Set(bits_count*2+1, bits_count+1)
    addr = 0b011
    training_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    addr = 0b111
    training_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    training_Dataset_Set.sort()
    
    test_Dataset_Set = Dataset_Set(bits_count*2+1, bits_count+1)
    addr = 0b000
    test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    addr = 0b011
    test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    addr = 0b100
    test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    addr = 0b111
    test_Dataset_Set.add_binary(addr,Dataset_Set.explain_as_full_adder(addr,1)[3])
    test_Dataset_Set.sort()
    if "random dataset" and False:
        training_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, max_amount=2)
        assert training_Dataset_Set.max_input_bits == 3
        # not sure. assert training_Dataset_Set.get_recommended_addr_FieldLength() == 3
        assert training_Dataset_Set.get_output_count() == 2
        
        test_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, max_amount=4)
        assert test_Dataset_Set.max_input_bits == 3
        assert test_Dataset_Set.get_output_count() == 2
        
        pass
    
    a_DatasetField_Set = DatasetField_Set(training_Dataset_Set)
    assert a_DatasetField_Set.get_input_count() == 3
    assert a_DatasetField_Set.get_output_count() == 2
    _result_tuple_bbsil = a_DatasetField_Set.valid(training_Dataset_Set)
    _perfect = _result_tuple_bbsil[1]
    assert _perfect
    
    _result_tuple_bbsil = a_DatasetField_Set.valid(test_Dataset_Set)
    _finished = _result_tuple_bbsil[0]
    assert _finished

    #print(list_of_total_and_error)
    total_diff_as_number = 0
    total_diff_bits = 0
    raw_test_dataset_list:list = test_Dataset_Set.get_as_int___check_btw()
    for _input_output in raw_test_dataset_list:
        lookup_result = a_DatasetField_Set.lookup(_input_output[0])
        
        diff_as_number = abs(lookup_result - _input_output[1])  
        total_diff_as_number += diff_as_number
        
        _xor_of_them = lookup_result^_input_output[1]
        diff_bits_count = count_ones(_xor_of_them)
        total_diff_bits += diff_bits_count
        pass
    
    '''
    about the dimention: 
    for every sample(people call it batch in deep learning), 
    ints:
    diff_bits, number_value(ground_truth, inferenced, diff, max_possible)
    _output_bit_count_per_sample is training_Dataset_Set.get_output_count()
    
    for all samples:
    total_diff_bits is a sum.
    _total_bits is raw_dataset_list.__len__()*bits_per_output
    total_diff_as_number is a sum.
    
    [below are returned.]    
    average:
    floats:
    avg_bit_wise_acc(1 - diff/total)
    #repeated in final result: avg_diff_over_max
    
    batch dim:
    ints:
    _training_sample_count is training_data.sample_count()
    _test_sample_count is test_data.sample_count()
    _max_possible_count is 1<<dataset.input_bit_count
    floats:
    referenced_acc = 1- ((_max_possible_count - _training_sample_count)/2)/max_possible_count
    
    result:
    avg_diff_as_number_over_max_possible_value is the avg of error.
    avg_bit_wise_acc - referenced_acc is the accuracy increase. The most important number.
    '''
    
    
    total_diff_as_number #ready 
    total_diff_bits #ready 
    _output_bit_count_per_sample = training_Dataset_Set.get_output_count()
    _total_bits = raw_test_dataset_list.__len__() * _output_bit_count_per_sample
        
    _max_possible_value = (1<<_output_bit_count_per_sample)-1
    
    avg_bit_wise_acc = 1 - total_diff_bits/_total_bits
    
    _training_sample_count = training_Dataset_Set.get_sample_count___maybe_unsafe()
    _test_sample_count = raw_test_dataset_list.__len__()
    _max_possible_sample_count = 1<<training_Dataset_Set.max_input_bits
    
    referenced_acc = 1- ((_max_possible_sample_count - _training_sample_count)/2.)/_max_possible_sample_count
    
    #result
    avg_diff_as_number_over_max_possible_value = (total_diff_as_number/_test_sample_count)/_max_possible_value
    aaaaaaaaaaaaaaaaaaaaaaa = avg_bit_wise_acc - referenced_acc 
    pass

if "prepare for the function.":
    #input zone.
    
    input_bit_count = 2

    #let's fix the random seed for this test.
    training_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, max_amount=5,              seed = 123)
    test_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, max_amount=11, proportion=0.7, seed = 321)
    
    
    #safety
    assert training_Dataset_Set.max_input_bits == test_Dataset_Set.max_input_bits 
    assert training_Dataset_Set.get_output_count() == test_Dataset_Set.get_output_count()
    
    a_DatasetField_Set = DatasetField_Set(training_Dataset_Set)
    _result_tuple_bbsil = a_DatasetField_Set.valid(training_Dataset_Set)
    _perfect = _result_tuple_bbsil[1]
    assert _perfect
    
    #redundent.
    _result_tuple_bbsil = a_DatasetField_Set.valid(test_Dataset_Set)
    finished = _result_tuple_bbsil[0]
    assert finished

    #real test.
    total_diff_as_number = 0
    total_diff_bits = 0
    raw_test_dataset_list = test_Dataset_Set.get_as_int___check_btw()
    for _input_output in raw_test_dataset_list:
        lookup_result = a_DatasetField_Set.lookup(_input_output[0])
        
        diff_as_number = abs(lookup_result - _input_output[1])  
        total_diff_as_number += diff_as_number
        
        _xor_of_them = lookup_result^_input_output[1]
        diff_bits_count = count_ones(_xor_of_them)
        total_diff_bits += diff_bits_count
        pass
    
    #get to the report
    #total_diff_as_number #ready 
    #total_diff_bits #ready 
    
    _training_sample_count = training_Dataset_Set.get_sample_count___maybe_unsafe()
    _test_sample_count = raw_test_dataset_list.__len__()
    assert raw_test_dataset_list.__len__() == test_Dataset_Set.get_sample_count___maybe_unsafe()
    
    _output_bit_count_per_sample = training_Dataset_Set.get_output_count()
    assert _output_bit_count_per_sample == input_bit_count+1
    _total_tested_bits = _test_sample_count * _output_bit_count_per_sample
        
    _max_possible_value = (1<<_output_bit_count_per_sample)-1
    
    avg_bit_wise_acc = 1 - total_diff_bits/_total_tested_bits
    
    _max_possible_sample_count = 1<<training_Dataset_Set.max_input_bits
    
    referenced_acc = 1- ((_max_possible_sample_count - _training_sample_count)/2.)/_max_possible_sample_count
    
    #result
    avg_diff_as_number = total_diff_as_number/_test_sample_count
    avg_diff_as_number_over_max_possible_value = avg_diff_as_number/_max_possible_value
    accuracy_gain = avg_bit_wise_acc - referenced_acc 
    pass

def accuracy_gain_test___add(training_Dataset_Set:Dataset_Set, test_Dataset_Set:Dataset_Set \
                )->tuple[numpy.ndarray,numpy.ndarray,float,float,float]:
    '''
    return accuracy_gain__ol, bitwise_acc__ol, referenced_acc__sc, avg_diff_as_number, avg_diff_as_number_over_max_possible_value
    
    >>> accuracy_gain__ol: numpy array [dim:output length in bit] If all the samples outside the training dataset are randomly decided, then this is 0..
    If the tool magically provides any extra accuracy for free, this is positive.
    
    >>> bitwise_acc__ol: numpy array [dim:output length in bit] average bitwise accuracy
    >>> referenced_acc__sc: calculated with the training_Dataset_Set sample count, and 1<<input bits count.
    
    >>> avg_diff_as_number: This value means something ONLY when the output bits is an binary integer.
    >>> avg_diff_as_number_over_max_possible_value: previous value divided by it's max possible value.
    The last 2 values implies the number is unsigned int.
    '''
    
    
    #safety
    assert training_Dataset_Set.max_input_bits == test_Dataset_Set.max_input_bits 
    assert training_Dataset_Set.get_output_count() == test_Dataset_Set.get_output_count()
    
    #train and check
    a_DatasetField_Set = DatasetField_Set(training_Dataset_Set)
    _result_tuple_bbsil = a_DatasetField_Set.valid(training_Dataset_Set)
    _perfect = _result_tuple_bbsil[1]
    assert _perfect
    
    #redundent.
    _result_tuple_bbsil = a_DatasetField_Set.valid(test_Dataset_Set)
    finished = _result_tuple_bbsil[0]
    assert finished#may not stable. But in most cases this will pass.

    #real test.
    #total_diff_bits = 0
    
    '''about the dimention info. 
    sc: SCalar, a single number.
    scfl: SCalar, but it's a FLag(std::bitset in cpp)
    ol: Output Length in bit. (test_Dataset_Set.get_output_count()).(it's output length, not office lady.)
    es: tEst_dataset Sample_count.
    rs: tRaining_dataset Sample_count
    '''
    bitwise_diff_count__ol:numpy.ndarray = numpy.zeros(shape=test_Dataset_Set.get_output_count(), dtype=numpy.uint32)
    total_diff_as_number__sc = 0
    raw_test_dataset__es:list[tuple[int, int]] = test_Dataset_Set.get_as_int___check_btw()
    for _input_output__2 in raw_test_dataset__es:
        # _input_output is 2 scalars.
        lookup_result__sc = a_DatasetField_Set.lookup(_input_output__2[0])
        
        diff_as_number__sc = abs(lookup_result__sc - _input_output__2[1])  
        total_diff_as_number__sc += diff_as_number__sc
        
        _xor_of_them__scfl = lookup_result__sc^_input_output__2[1]
        _neg_index = -1
        while _xor_of_them__scfl>0:
            if _xor_of_them__scfl&0b1:
                bitwise_diff_count__ol[_neg_index] +=1
                pass
            #tail 
            _xor_of_them__scfl >>=1
            _neg_index -=1
            pass
        ''' old code  
        diff_bits_count = count_ones(_xor_of_them)
        total_diff_bits += diff_bits_count'''
        pass
    
    #get to the report
    #total_diff_as_number__sc #ready 
    #total_diff_bits #ready deleted.
    #bitwise_diff_count___nparr #ready
    
    _training_sample_count__the_rs__sc:int = training_Dataset_Set.get_sample_count___maybe_unsafe()
    _test_sample_count__the_es__sc:int = raw_test_dataset__es.__len__()
    assert raw_test_dataset__es.__len__() == test_Dataset_Set.get_sample_count___maybe_unsafe()
    
    #1w 继续补维度信息。
    
    _output_bit_count_per_sample__the_ol__sc = training_Dataset_Set.get_output_count()
    assert _output_bit_count_per_sample__the_ol__sc == input_bit_count+1
    #_total_tested_bits = _test_sample_count__the_es__sc * _output_bit_count_per_sample__the_ol__sc
        
    _max_possible_value = (1<<_output_bit_count_per_sample__the_ol__sc)-1
    
    #old code .avg_bit_wise_acc = 1 - total_diff_bits/_total_tested_bits
    bitwise_acc__ol = 1 - bitwise_diff_count__ol/_test_sample_count__the_es__sc
    _max_possible_sample_count__sc = 1<<training_Dataset_Set.max_input_bits
    referenced_acc__sc = 1- ((_max_possible_sample_count__sc - _training_sample_count__the_rs__sc)/2.0)/float(_max_possible_sample_count__sc)
    assert referenced_acc__sc>=0.5
    assert referenced_acc__sc<1.
    accuracy_gain__ol = bitwise_acc__ol - referenced_acc__sc 
    assert accuracy_gain__ol.shape == (_output_bit_count_per_sample__the_ol__sc,)
    
    #add test only.
    avg_diff_as_number = total_diff_as_number__sc/_test_sample_count__the_es__sc
    avg_diff_as_number_over_max_possible_value = avg_diff_as_number/_max_possible_value
    
    return accuracy_gain__ol, bitwise_acc__ol, referenced_acc__sc, avg_diff_as_number, avg_diff_as_number_over_max_possible_value




if "test the function." and True:
    input_bit_count = 5
    #let's fix the random seed for this test.
    training_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, max_amount=501, proportion=0.7)
    test_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(    input_bit_count, max_amount=111, proportion=0.7)
    assert training_Dataset_Set.get_sample_count___maybe_unsafe() == 501
    assert test_Dataset_Set.get_sample_count___maybe_unsafe() == 111
    
    accuracy_gain__ol, \
        bitwise_acc__ol, \
            referenced_acc__sc, \
                avg_diff_as_number, \
                    avg_diff_as_number_over_max_possible_value \
        = accuracy_gain_test___add(training_Dataset_Set, test_Dataset_Set)
    over_all_acc_gain = accuracy_gain__ol.mean()
    over_all_bitwise_acc = bitwise_acc__ol.mean()
    
    
    
    
    
    
    
    input_bit_count = 3
    #let's fix the random seed for this test.
    training_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, max_amount=11, proportion=0.7, seed = 123)
    test_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(    input_bit_count, max_amount=13, proportion=0.7, seed = 321)
    assert training_Dataset_Set.get_sample_count___maybe_unsafe() == 11
    assert test_Dataset_Set.get_sample_count___maybe_unsafe() == 13
    
    accuracy_gain__ol, \
        bitwise_acc__ol, \
            referenced_acc__sc, \
                avg_diff_as_number, \
                    avg_diff_as_number_over_max_possible_value \
        = accuracy_gain_test___add(training_Dataset_Set, test_Dataset_Set)
    pass
    
    
    
if "method: accuracy_gain_test___with_uint_report\n\n   slow" and True:
    #file name
    _time = datetime.now()
    _time_str = _time.isoformat(sep=" ")
    _time_str = _time_str[0:19]
    _time_str = _time_str.replace(":","-")
    _file_name = f"acc gain test\\acc gain test result {_time_str}.txt"
    with open(_file_name, mode = "a", encoding="utf-8") as file:
        file.write("method: accuracy_gain_test___with_uint_report\n\n")
        file.write(f"{_time_str}\n\n")
        pass#open
    
    for input_bit_count in range(5,16):
        # _test_time
        _test_time = 2
        if input_bit_count<=10:
            _test_time = 3
            pass
        if input_bit_count<=8:
            _test_time = 4
            pass
        if input_bit_count<=5:
            _test_time = 5
            pass
        with open(_file_name, mode = "a", encoding="utf-8") as file:
            _repeating_str_previous = f"{input_bit_count-1}  "*18
            file.write(f"end of   {_repeating_str_previous}\n")
            _repeating_str_here = f"{input_bit_count}  "*18
            file.write(f"start of {_repeating_str_here}\n")
            file.write(f"test time {_test_time}\n\n")
            pass
            
        #init
        last_training_sample_count = -1
        working = False
        
        for training_proportion in [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5]:
            #checks if the sample is too few.
            #checks the training data
            proportion, count, the_proportion_is_the_limit = Dataset_Set. \
                get_full_adder_testset_partly___sample_count_detect(input_bit_count, proportion=training_proportion, \
                    max_amount=30_000)
            training_sample_count = count
            if training_sample_count<10:
                continue
            if training_sample_count == last_training_sample_count:
                #either the proportion is too small or too big
                if working:#too big. next input_bit_count
                    break
                else:#too small, next proportion.
                    continue
            else:
                working = True
                last_training_sample_count = training_sample_count
                pass
            #checks the test data
            proportion, count, the_proportion_is_the_limit = Dataset_Set. \
                get_full_adder_testset_partly___sample_count_detect(input_bit_count, proportion=0.79, \
                    max_amount=10_000)
            test_sample_count = count
            if test_sample_count<25:
                continue
            
            '''loop split trick again. 
            this loop is splitted into 2 parts.'''
            '''part 1'''
            #datasets
            training_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, proportion=training_proportion, \
                    max_amount=30_000)
            assert training_Dataset_Set.get_sample_count___maybe_unsafe() == training_sample_count
            test_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, proportion=0.79, \
                    max_amount=10_000)
            assert test_Dataset_Set.get_sample_count___maybe_unsafe() == test_sample_count
            #training and valid.
            accuracy_gain__ol, \
                bitwise_acc__ol, \
                    referenced_acc__sc, \
                        avg_diff_as_number, \
                            avg_diff_as_number_over_max_possible_value \
                = accuracy_gain_test___add(training_Dataset_Set, test_Dataset_Set)
            #over_all_acc_gain = accuracy_gain__ol.mean()
            #over_all_bitwise_acc = bitwise_acc__ol.mean()
            '''part 2'''
            for _ in range(1, _test_time):
                #datasets
                training_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, proportion=training_proportion, \
                        max_amount=30_000)
                test_Dataset_Set = Dataset_Set.get_full_adder_testset_partly(input_bit_count, proportion=0.79, \
                        max_amount=10_000)
                #training and valid.
                in_loop__accuracy_gain__ol, \
                    in_loop__bitwise_acc__ol, \
                        in_loop__referenced_acc__sc, \
                            in_loop__avg_diff_as_number, \
                                in_loop__avg_diff_as_number_over_max_possible_value \
                    = accuracy_gain_test___add(training_Dataset_Set, test_Dataset_Set)
                
                accuracy_gain__ol += in_loop__accuracy_gain__ol
                bitwise_acc__ol += in_loop__bitwise_acc__ol
                assert referenced_acc__sc == in_loop__referenced_acc__sc
                avg_diff_as_number += in_loop__avg_diff_as_number
                avg_diff_as_number_over_max_possible_value += in_loop__avg_diff_as_number_over_max_possible_value
                
                #over_all_acc_gain += in_loop__accuracy_gain__ol.mean()
                #over_all_bitwise_acc += in_loop__bitwise_acc__ol.mean()
                pass#for in range
            
            #divides by the _test_time
            accuracy_gain__ol /= _test_time
            bitwise_acc__ol /= _test_time
            assert referenced_acc__sc >= 0.5 and referenced_acc__sc < 1.
            avg_diff_as_number /= _test_time
            avg_diff_as_number_over_max_possible_value /= _test_time
                
            over_all_acc_gain = accuracy_gain__ol.mean()
            over_all_bitwise_acc = bitwise_acc__ol.mean()
            
            
            
            #log out.
            with open(_file_name, mode = "a", encoding="utf-8") as file:
                file.write(f"input_bit_count: {input_bit_count}\n")
                file.write(f"test time: {_test_time}\n")
                _actual_training_proportion = training_sample_count/(1<<(input_bit_count*2+1))
                file.write(f"training_proportion: {_actual_training_proportion:.5f} ({training_proportion})\n")
                file.write(f"training_sample_count: {training_sample_count}\n")
                file.write(f"test_sample_count: {test_sample_count}\n")
                file.write(f"-  -  -  -  -  -  -  -  -  -  -  -  \n")
                
                file.write(f"               avg ACC GAIN: {over_all_acc_gain}")
                if over_all_acc_gain >0:
                    file.write(f"\n")
                    pass
                else:
                    file.write(f", BAD! BAD! BAD! BAD!!!!!!!!!!!!!!!\n")
                    pass
                
                file.write(f"      avg acc: {over_all_bitwise_acc}\n")
                
                file.write(f"bitwise ACC GAIN: {accuracy_gain__ol}\n")
                file.write(f"bitwise ACC: {bitwise_acc__ol}\n")
                
                _lets_calc_ref_acc_again = training_proportion/2.+0.5
                file.write(f"referenced_acc: {referenced_acc__sc:.4f}({_lets_calc_ref_acc_again:.4f})\n")
                file.write(f"-  -  -  -  -  -  -  -  -  -  -  -  \n")
                
                file.write(f"avg_diff_as_number: {avg_diff_as_number:.4f}\n")
                file.write(f"avg_diff_as_number_over_max_possible_value: {avg_diff_as_number_over_max_possible_value:.5f}\n")
                
                file.write(f"       max_possible_value: {(1<<_output_bit_count_per_sample)-1}\n")
                file.write("\n\n")
                pass# open
            pass# for training_proportion in range
        pass# for input_bit_count in range(3,22):
                
    pass
    
    
from array import array

input_file = open('foo', 'r')
float_array = array('d')
float_array.fromstring(input_file.read())

print float_array

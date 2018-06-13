from array import array
output_file = open('foo', 'wb')
float_array = array('d', [3.1413333332244434] * 100)
float_array.tofile(output_file)
output_file.close()


input_file = open('foo', 'r')
float_array = array('d')
float_array.fromstring(input_file.read())

print float_array

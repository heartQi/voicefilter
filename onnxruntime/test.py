import numpy
import onnxruntime

def read_dump(path):
  file = open(path, "r")
  data = []
  for line in file:
    start = line.find("(")
    end = line.find(")")
    data.append(float(line[start+1:end]))
  return numpy.array(data, dtype=numpy.float32)

model_dir ="./model"
model=model_dir+"/voice_filter_sim.onnx"

input = read_dump("./input.data")
input = numpy.expand_dims(input, axis=(0, 1))

dvec = read_dump("./dvec.data")
dvec = numpy.expand_dims(dvec, axis=0)
print(dvec)

session = onnxruntime.InferenceSession(model, None)
input1_name = session.get_inputs()[0].name
input2_name = session.get_inputs()[1].name
output_name = session.get_outputs()[0].name
print(input1_name)
print(input2_name)
print(output_name)

result = session.run([output_name], {input1_name: input, input2_name: dvec})
print(numpy.array(result))

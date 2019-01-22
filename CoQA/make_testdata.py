import json
with open("coqa-dev-v1.0.json", "r", encoding="utf8") as f:
    dev_data = json.load(f)
test_one=dev_data["data"][0]

### STORY
test_one["story"] = "My name is Guotong. Guotong favourite color is black. Guotong do not live alone. Guotong lives in beijing"
test_one["questions"] = []
test_one["answers"] = []

### QUESTION

one_question = {}
one_question["input_text"]="What is Guotong favourite color?"
one_question["turn_id"]=1
test_one["questions"].append(one_question)

one_question = {}
one_question["input_text"]="do he live alone?"
one_question["turn_id"]=2
test_one["questions"].append(one_question)

one_question = {}
one_question["input_text"]="where is he living?"
one_question["turn_id"]=3
test_one["questions"].append(one_question)

### ANSWER

one_answer = {}
one_answer["span_start"] = 0
one_answer["span_end"] = 0
one_answer["span_text"] = "Guotong favourite color is black"
one_answer["input_text"] = "black"
one_answer["turn_id"] = 1
test_one["answers"].append(one_answer)

one_answer = {}
one_answer["span_start"] = 0
one_answer["span_end"] = 0
one_answer["span_text"] = "Guotong do not live alone"
one_answer["input_text"] = "no"
one_answer["turn_id"] = 2
test_one["answers"].append(one_answer)

one_answer = {}
one_answer["span_start"] = 0
one_answer["span_end"] = 0
one_answer["span_text"] = "Guotong lives in beijing"
one_answer["input_text"] = "beijing"
one_answer["turn_id"] = 3
test_one["answers"].append(one_answer)

###

test_one["additional_answers"] = {}

dev_data["data"] = [test_one]

with open("coqa-test-v1.0.json", "w", encoding="utf8") as f:
  json.dump(dev_data,f)
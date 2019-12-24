import os
import time

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print("req send 1", current_time)

os.system('python translate.py -model available_models/model_en-hi_exp-5.10_2019-12-20-model_step_150000.pt -src test2.txt -output test2_o.txt -replace_unk -verbose -gpu 1')
print("hhherere*********************************************888")
os.system('python translate.py -model available_models/model_en-hi_inc_exp-5.10_2019-12-22-model_step_250000.pt -src test1.txt -output test1_o.txt -replace_unk -verbose -gpu 0')
print("res 2 received ", current_time)
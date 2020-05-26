Interactive Translation

Steps involved:
1. Convert the trained model .pt file into binary file along with source and target vocabulary
   The "interactive-model-convert" api expects following input:
   1. inp_model_path - path of model(.pt) file which needs to be converted
   2. out_dir - output path where binary model file and vocabulary are saved

2. The converted model has to be updated in available_models/interactive_models/iconf.json file along with model id and the path where it
   is saved(i.e out_dir)

3. Make corresponding model changes in interactive_translation function and deploy
4. The interactive apis can be accessed by running interactive_server.py   
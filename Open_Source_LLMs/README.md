# LLM-Research Open Source Experiments
 
Begin by ensuring you have the appropriate packages installed (run `pip install -r requirements.txt`).

<!-- Modify the files as follows: -->

Generate Explanations
- `generate_sentences.py`
    - This file will generate a subset of sentences from the SST dataset and store this subset into `sentences.pickle`
- `generate_model_expl.py`
    - use `python3 generate_model_expl.py -h` to view helpful info
    - use `-pe` flag to set the generation mode to predict and explain, otherwise the generation mode is explain then predict
    - use `-mistral` to use the mistral model (makesure to install "mistralai/Mistral-7B-Instruct-v0.2")
    - use `-llama` to use the llama model (makesure to install meta-llama/Meta-Llama-3-8B-Instruct)
    - rename the model name in `generate_model_expl.py` to use any other models
- `generate_lime_expl.py`
    - use `python3 generate_lime_expl.py -h` to view helpful info
    - use `-pe` flag to set the generation mode to predict and explain, otherwise the generation mode is explain then predict
    - use `-mistral` to use the mistral model (makesure to install "mistralai/Mistral-7B-Instruct-v0.2")
    - use `-llama` to use the llama model (makesure to install meta-llama/Meta-Llama-3-8B-Instruct)
    - rename the model name in `generate_lime_expl.py` to use any other models
- `generate_occlusion_expl.py`
    - use `python3 generate_occlusion_expl.py -h` to view helpful info
    - use `-mistral` to use the mistral model (makesure to install "mistralai/Mistral-7B-Instruct-v0.2")
    - use `-llama` to use the llama model (makesure to install meta-llama/Meta-Llama-3-8B-Instruct)
    - rename the model name in `generate_lime_expl.py` to use any other models

Evaluate Explanations:
- `process.py`
    - this file will run evaluation metrics (comprehensiveness, sufficiency, deletion rank correlation ......)
    - use `python3 process.py -h` to view helpful info
    - arguments include setting the response filename, output file name, label file name, generation mode, explanation mode (occlusion, lime), and language models

- `disagreement.py`
    - this file will measure the disagreement among explanations being generated from different methods
    

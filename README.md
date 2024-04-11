# LLM-Research
 
Begin by ensuring you have the appropriate packages installed (run pip install -r $requirements.txt$).

Modify the files as follows:

LIME
- generate_lime_expl.py
    - adjust prompts (PE, EP)
    - change model to desired
    - ensure sentences.pickle exists (if not, first run generate_model_expl.py. Optionally can comment out the model generation code to be left with sentence.pickle)
    - adjust batch size as well as start and end (sentence index)
- process.py
    - change prompt to same as generate_lime_expl.py
    - change response filename to the one in generate_lime_expl.py
    - ensure accuracy is commented out
    - set PE to the appropriate value based on prompt
    - Rename print statements/variables based on the current model


Model Explanations:
- generate_model_expl.py
    - adjust prompts (PE, EP)
    - change model to desired
    - change response and label filenames (line 82 & 84)
- process.py
    - change prompt to same as generate_model_expl.py
    - change response filename to the one in generate_model_expl.py
    - ensure accuracy is not commented
    - set PE to the appropriate value based on prompt
    - Rename print statements/variables based on the current model
import os, sys
from spacy_llm.util import assemble
from usefull import reassembleDataset
import pandas as pd
import json

# set your open-AI api key as environmental variable :
openAiKey = ""
with open("open-ai-api-key.txt", "r") as keyFile:
    openAiKey = keyFile.readlines()[0].strip().replace("\n","")
# os.environ["OPENAI_API_KEY"] = openAiKey 

# load the model settings from the config file
nlp = assemble("fewshot.cfg")
doc = nlp("Begin by preparing the flavorful meat sauce. In a large pan over medium heat, sauté finely chopped onions in olive oil until translucent. Add minced garlic and ground beef, cooking until the beef is browned. Break apart the meat with a spoon for an even texture.")

# print the llm output
print([(ent.text, ent.label_) for ent in doc.ents])



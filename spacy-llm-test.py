import os, sys
from spacy_llm.util import assemble
from usefull import reassembleDataset
import pandas as pd
import json

# set your open-AI api key as environmental variable :
# os.environ["OPENAI_API_KEY"] = "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"

# load the model settings from the config file
nlp = assemble("fewshot.cfg")
doc = nlp("Begin by preparing the flavorful meat sauce. In a large pan over medium heat, sauté finely chopped onions in olive oil until translucent. Add minced garlic and ground beef, cooking until the beef is browned. Break apart the meat with a spoon for an even texture.")

# print the llm output
print([(ent.text, ent.label_) for ent in doc.ents])



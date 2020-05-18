# # Named Entity Recognition
# 
# As Parts of Speach is important in any given sentence, so is a Named Entity like Name, Place, Organization, Time, Currency, etc. We have a module in Spacy that can provide us with these information too.

#import spacy and load the language model
import spacy
lingua = spacy.load('en_core_web_sm')

#create a function to read file contents to a spacy document.
def read_file(fp):
    with open(fp) as f:
        doc = lingua(f.read())
    return doc

# Load the file content and save them into the doc object
doc = read_file('Large_Text.text')

#Let us check a random sentence
print(doc[:21])

# Write a function to display basic entity info:
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')

#Call the above function to print the Named Entity
show_ents(doc[:21])

# Let us import Random module of Python and generate a random integer to check the NEs.
import random
rand_idx = random.randint(0,len(doc))

#Print the named entity in the randomly selected sentence
show_ents(doc[:rand_idx])

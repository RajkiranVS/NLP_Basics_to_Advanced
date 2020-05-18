
# Parts of Speach plays a major role in any Natural Language it sets the basic rule of sentence construction. To process Natural Language we need some modules that can identify the parts of speach correctly. Luckily, we have a module in Spacy that can perform this task

# Load the spacy model and language model object
import spacy
lingua = spacy.load('en_core_web_sm')

#create a function to read file contents to a spacy document.
def read_file(fp):
    with open(fp) as f:
        doc = lingua(f.read())
    return doc


# Lopad the file object and store the tokens in the doc object using the above function
doc = read_file('Large_Text.text')

#Let us check the parts of speach in a randomly selected sentence
for token in doc[:21]:
    print(token.text,':',token.pos_)

#Print the POS in proper format
for token in doc[:21]:
    print(f'{token.text:12}......> {token.pos_:20}{spacy.explain(token.pos_)}')

#Print the DET in proper format
for token in doc[:21]:
    print(f'{token.text:12}......> {token.dep_:20}{spacy.explain(token.dep_)}')


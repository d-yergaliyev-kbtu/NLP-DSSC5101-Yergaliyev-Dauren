import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

text = "The Kazakh dombyra has frets and is played by strumming with the hand or plucking each string individually, with an occasional tap on the main surface of the instrument."

doc = nlp(text)

print("Named Entities, Phrases, and Concepts:")
for entity in doc.ents:
    print(f"{entity.text} - {entity.label_} ({spacy.explain(entity.label_)})")

displacy.render(doc, style="ent")

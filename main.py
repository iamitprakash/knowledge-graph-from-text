import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Perform basic preprocessing
    doc = nlp(text)
    return doc

def extract_entities_and_relations(doc):
    entities = []
    relations = []
    
    for entity in doc.ents:
        entities.append((entity.text, entity.label_))
    
    for token in doc:
        if token.dep_ in ("nsubj", "dobj"):
            subject = token.text
            relation = token.head.text
            object_ = [w for w in token.head.children if w.dep_ == "dobj"]
            object_ = object_[0].text if object_ else ""
            if subject and relation and object_:
                relations.append((subject, relation, object_))
    
    return entities, relations

def build_graph(entities, relations):
    G = nx.Graph()
    
    for entity, entity_type in entities:
        G.add_node(entity, entity_type=entity_type)
    
    for subject, relation, object_ in relations:
        G.add_edge(subject, object_, relation=relation)
    
    return G

def visualize_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Knowledge Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
text = "Apple Inc. was founded by Steve Jobs in Cupertino. The company produces iPhones."
doc = preprocess_text(text)
entities, relations = extract_entities_and_relations(doc)
graph = build_graph(entities, relations)
visualize_graph(graph)

# Print extracted information
print("Entities:", entities)
print("Relations:", relations)

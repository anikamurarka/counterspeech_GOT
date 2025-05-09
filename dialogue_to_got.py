import argparse
import pickle
import string
from pathlib import Path

import spacy
import numpy as np
import stanza
from stanza.server import CoreNLPClient
from tqdm import tqdm

from train_utils.utils_data import load_data

stanza.install_corenlp()
nlp = spacy.load('en_core_web_sm')
punctuation_marks = string.punctuation
letters = "([A-Za-z])"
title_prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
name_suffixes = "(Inc|Ltd|Jr|Sr|Co)"
sentence_beginners = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
abbrev = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
web_domains = "[.](com|net|org|io|gov|me|edu)"
node_limit = 100


def resolve_references(text):
    """Simple coreference resolution function"""
    document = nlp(text)
    # Simplified implementation that returns empty list
    # Graph extraction will work without coreference resolution
    return []


def get_relation_triples(doc):
    extracted_triples = []
    for sentence in doc.sentence:
        for triple in sentence.openieTriple:
            subj = getattr(triple, 'subject')
            rel = getattr(triple, 'relation')
            obj = getattr(triple, 'object')

            extracted_triples.append({'subject': subj, 'relation': rel, 'object': obj})
    return extracted_triples


def consolidate_triples(annotation_result, reference_clusters):
    extracted_triples = get_relation_triples(annotation_result)

    unique_triples = []
    for i in range(0, len(extracted_triples)):
        current = extracted_triples[i]
        current_subj = current['subject'].lower()
        current_rel = current['relation'].lower()
        current_obj = current['object'].lower()

        for cluster in reference_clusters:
            mention_texts = [mention.text.lower() for mention in cluster.mentions]
            if current_subj in mention_texts:
                current_subj = cluster.main.text.lower()
            if current_obj in mention_texts:
                current_obj = cluster.main.text.lower()

        if len(unique_triples) == 0:
            unique_triples.append([current_subj, current_rel, current_obj])
        else:
            is_duplicate = False
            for j in range(0, len(unique_triples)):
                # Keep the longer version when entities match
                if unique_triples[j][0] == current_subj and unique_triples[j][1] == current_rel:
                    if len(current_obj) > len(unique_triples[j][2]):
                        unique_triples[j][2] = current_obj
                    is_duplicate = True

                elif unique_triples[j][0] == current_subj and unique_triples[j][2] == current_obj:
                    if len(current_rel) > len(unique_triples[j][1]):
                        unique_triples[j][1] = current_rel
                    is_duplicate = True

                elif unique_triples[j][2] == current_obj and unique_triples[j][1] == current_rel:
                    if len(current_subj) > len(unique_triples[j][0]):
                        unique_triples[j][0] = current_subj
                    is_duplicate = True

            if not is_duplicate:
                # New triple, add to collection
                unique_triples.append([current_subj, current_rel, current_obj])

    return unique_triples


def generate_knowledge_graph(context, node_limit, nlp_client):
    """Generate knowledge graph from text

    Args:
        context (string): text to analyze
        node_limit (int): maximum number of nodes
        nlp_client: CoreNLP client

    Returns:
        formatted_input (list): formatted text representation of graph
        adjacency_matrix (numpy array): graph structure 
    """
    context = context.replace("\n", " ")
    reference_clusters = resolve_references(context)

    context = context.replace("\n", " ")
    annotation_result = nlp_client.annotate(context)
    triples = consolidate_triples(annotation_result, reference_clusters)

    formatted_input = []

    entity_to_id = {}
    id_to_entity = {}
    adjacency_matrix = np.zeros([node_limit, node_limit])
    node_count = 0
    
    if len(triples) == 0:
        formatted_input.append('<pad>')
    else:
        text_representation = ' <s> '
        for triple in triples:
            # Process subject
            if triple[0] not in entity_to_id:
                entity_to_id[triple[0]] = node_count
                id_to_entity[node_count] = triple[0]
                if node_count < node_limit:
                    if text_representation == ' <s> ':
                        text_representation = text_representation + triple[0]
                    else:
                        text_representation = text_representation + ' </s> <s> ' + triple[0]
                    node_count += 1
                else:
                    break
                    
            # Process relation
            if triple[1] not in entity_to_id:
                entity_to_id[triple[1]] = node_count
                id_to_entity[node_count] = triple[1]
                if node_count < node_limit:
                    if text_representation == ' <s> ':
                        text_representation = text_representation + triple[1]
                    else:
                        text_representation = text_representation + ' </s> <s> ' + triple[1]
                    node_count += 1
                else:
                    break
                    
            # Process object
            if triple[2] not in entity_to_id:
                entity_to_id[triple[2]] = node_count
                id_to_entity[node_count] = triple[2]
                if node_count < node_limit:
                    if text_representation == ' <s> ':
                        text_representation = text_representation + triple[2]
                    else:
                        text_representation = text_representation + ' </s> <s> ' + triple[2]
                    node_count += 1
                else:
                    break

            # Update adjacency matrix
            subj_id = entity_to_id[triple[0]]
            rel_id = entity_to_id[triple[1]]
            obj_id = entity_to_id[triple[2]]
            
            # Self-connections
            adjacency_matrix[subj_id][subj_id] = 1
            adjacency_matrix[rel_id][rel_id] = 1
            adjacency_matrix[obj_id][obj_id] = 1
            
            # Subject-relation connections
            adjacency_matrix[subj_id][rel_id] = 1
            adjacency_matrix[rel_id][subj_id] = 1
            
            # Relation-object connections
            adjacency_matrix[rel_id][obj_id] = 1
            adjacency_matrix[obj_id][rel_id] = 1

        formatted_input.append(text_representation)

    return formatted_input, adjacency_matrix


def create_output_directory(args, split):
    # Create directory for output
    output_path = Path(args.output_dir) / f"{split}/"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def store_graph_data(input_text_list, adjacency_matrix_list, output_path, args):
    input_text_file_path = output_path / args.input_text_file
    with open(input_text_file_path, 'wb') as f:
        pickle.dump(input_text_list, f)

    adjacency_matrix_file_path = output_path / args.adj_matrix_file
    with open(adjacency_matrix_file_path, 'wb') as f:
        pickle.dump(adjacency_matrix_list, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='DIALOCONAN')
    parser.add_argument('--splits', nargs="+", default=["train", "dev", "test"])  # "mini-test"])
    parser.add_argument('--output_dir', type=str, default='data/DIALOCONAN/got/')
    parser.add_argument('--input_text_file', type=str, default='mc_input_text.pkl')
    parser.add_argument('--adj_matrix_file', type=str, default='mc_adj_matrix.pkl')
    parser.add_argument('--exclude_context', action='store_true', help='remove dialogue history from the prompt')

    args = parser.parse_args()
    return args


def main(args):
    for split in args.splits:
        # Create directories
        output_path = create_output_directory(args, split)

        # Read data
        dialogues = load_data(args, split)

        # Analyze
        input_text_list, adjacency_matrix_list = [], []
        with CoreNLPClient(annotators=["ner", "openie", "coref"], memory='4G',
                           endpoint='http://localhost:2727', be_quiet=True) as client:

            for dialogue in tqdm(dialogues):
                dialogue_history = dialogue["dialogue_history"]
                hate_speech = dialogue["hate_speech"]

                if args.exclude_context:
                    context_text = f"{hate_speech}"
                else:
                    context_text = f"{dialogue_history}\n{hate_speech}"

                input_text, adjacency_matrix = generate_knowledge_graph(context_text, node_limit, client)
                input_text_list.append(input_text)
                adjacency_matrix_list.append(adjacency_matrix)

            client.stop()

        # Save data
        store_graph_data(input_text_list, adjacency_matrix_list, output_path, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)

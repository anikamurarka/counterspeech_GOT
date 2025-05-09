import streamlit as st
import torch
import os
import numpy as np
import json
import tempfile
from transformers import AutoTokenizer
from train_utils.model import T5GenerationWithGraph
from train_utils.utils_prompt import postprocess_text
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import time
from train_utils.dataset import DialoconanDatasetWithGraph
from dialogue_to_got import store_graph_data
from train_utils.utils_prompt import build_train_pair_dialoconan
import pickle
import uuid
import tqdm
import requests
from dialogue_to_got import generate_knowledge_graph, node_limit
from stanza.server import CoreNLPClient
# Set page config
st.set_page_config(page_title="Counterspeech Generation", page_icon="💬", layout="wide")

@st.cache_resource
def load_model(model_path):
    """Load the trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-base")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
    vocabulary = tokenizer.get_vocab()
    special_token_id = vocabulary["<s>"]
    
    # Load the custom model with graph
    model = T5GenerationWithGraph.from_pretrained(model_path, s_token_id=special_token_id)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    return model, tokenizer
def create_formatted_dialogue(dialogue):
    # Exclude the final hate speech and counter-narrative entries
    dialogue_content = dialogue[:-2]

    # Create formatted utterances and combine them
    formatted_utterances = [f"{entry['type']}: {entry['text']}" for idx, entry in dialogue_content.iterrows()]
    combined_dialogue = "\n".join(formatted_utterances)

    return combined_dialogue

def organize_dialogue_data(grouped_dialogues):
    organized_data = []
    # Convert to pandas DataFrame for better inspection
    import pandas as pd
    if not isinstance(grouped_dialogues, pd.DataFrame):
        grouped_dialogues = pd.DataFrame(grouped_dialogues)
        # print(df)
    else:
        print(grouped_dialogues["text"])
    # for dialogue_content in grouped_dialogues:
    if len(grouped_dialogues) > 1:
        entry = {
                # "dialogue_id": dialogue_id[0],
                # "target": dialogue_content["TARGET"].unique()[0],

            "hate_speech": grouped_dialogues["text"].values[-2],
            "counter_narrative": grouped_dialogues["text"].values[-1],
            "dialogue_history": create_formatted_dialogue(grouped_dialogues)
        }
        organized_data.append(entry)
    else:
        entry = {
            "hate_speech": grouped_dialogues["text"].values[0],
            "counter_narrative": "",
            "dialogue_history": create_formatted_dialogue(grouped_dialogues)
        }
        organized_data.append(entry)
    return organized_data

def process_conversation_with_got(conversation_history):
    """Process the conversation through the GOT pipeline."""
    # Step 2: Use the same GOT processing as in pipeline.sh
    
    
    # Create a temporary output directory
    input_text_list, adjacency_matrix_list = [], []
    
    # Function to check if CoreNLP server is already running on given port
    def is_server_running(port):
        try:
            # Try to connect to the server's status endpoint
            response = requests.get(f"http://localhost:{port}", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    # Check if server is already running on default port 2727
    default_port = 2727
    start_server = not is_server_running(default_port)
    print("__________________________", start_server)
    try:
        # Create client with appropriate start_server parameter
        if start_server:
            client = CoreNLPClient(annotators=["ner", "openie", "coref"], 
                               memory='4G',
                               endpoint=f'http://localhost:{default_port}', 
                               be_quiet=True,
                              )  # Only start server if needed
            
        # Extract the hate speech from the conversation history
        print(conversation_history)
        for dialogue in conversation_history:
            dialogue_history = dialogue["dialogue_history"]
            hate_speech = dialogue["hate_speech"]

            context_text = f"{dialogue_history}\n{hate_speech}"

            input_text, adjacency_matrix = generate_knowledge_graph(context_text, node_limit, client)
            input_text_list.append(input_text)
            adjacency_matrix_list.append(adjacency_matrix)
        
        # Don't stop the client if we didn't start it
        # if start_server:
        # Force stop the client regardless of any errors
        try:
            client.stop()
        except Exception as e:
            print(f"Error stopping CoreNLP client: {str(e)}")
            # Force terminate if normal stop fails
            if hasattr(client, 'server'):
                try:
                    client.server.kill()
                except:
                    pass
            
        return input_text_list, adjacency_matrix_list
    except Exception as e:
        st.error(f"Error with CoreNLP server: {str(e)}")
        # Return empty lists as fallback
        return [], []

def generate_response(model, tokenizer, conversation_history, max_length=256):
    organized_data = organize_dialogue_data(conversation_history)
    input_text_list, adjacency_matrix_list = process_conversation_with_got(organized_data)
    dataset = []
    target_text = []
    source_text = []
    # organized_data = organize_dialogue_data(conversation_history)
    print(organized_data)
    print(input_text_list)
    print(adjacency_matrix_list)
    for idx, example in enumerate(organized_data):
        prompt, target = build_train_pair_dialoconan(example)
        source_text.append(prompt)
        target_text.append(target)
        

        # Normalize whitespace
        # source_text = " ".join(source_text.split())
        # target_text = " ".join(target_text.split())
        source_text = str(source_text[idx])
        target_text = str(target_text[idx])

        # Get graph data for the current example
        graph_nodes = input_text_list[idx]
        graph_matrix = torch.tensor(adjacency_matrix_list[idx])

        # Tokenize source text
        source_encoding = tokenizer.batch_encode_plus(
            [source_text],
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Tokenize target text
        target_encoding = tokenizer.batch_encode_plus(
            [target_text],
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Tokenize graph node text
        graph_encoding = tokenizer.batch_encode_plus(
            graph_nodes,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract tensors from encodings
        source_ids = source_encoding["input_ids"].squeeze()
        source_mask = source_encoding["attention_mask"].squeeze()
        target_ids = target_encoding["input_ids"].squeeze().tolist()
        graph_ids = graph_encoding["input_ids"].squeeze()
        graph_mask = graph_encoding["attention_mask"].squeeze()

        dataset.append(  {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            "got_adj_matrix": graph_matrix,
            "got_input_ids": graph_ids,
            "got_mask": graph_mask,
        })
    # model = "experiments/DIALOCONAN/declare-lab-flan-alpaca-base_lr5e-05_bs8_op256_ep20_2025-05-08-23-03"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
    vocabulary = tokenizer.get_vocab()
    special_token_id = vocabulary["<s>"]
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    # dataset = DialoconanDatasetWithGraph(conversation_history, "dev", tokenizer, 1024, 1024, None)
    model = T5GenerationWithGraph.from_pretrained(model, s_token_id=special_token_id)
    model.resize_token_embeddings(len(tokenizer))
    eval_args = Seq2SeqTrainingArguments(output_dir="results",
                                         do_train=False,
                                         do_eval=True,
                                         evaluation_strategy="no",
                                         logging_strategy='no',
                                         logging_steps='no',
                                         save_strategy='no',
                                         eval_steps='no',
                                         save_steps='no',
                                         save_total_limit='no',
                                         learning_rate=0.0001,
                                         eval_accumulation_steps=500,
                                         per_device_train_batch_size=1,
                                         per_device_eval_batch_size=1,
                                         weight_decay=0.01,
                                         num_train_epochs=1,
                                         metric_for_best_model="rougeL",
                                         predict_with_generate=True,
                                         generation_max_length=max_length,
                                         load_best_model_at_end=True,
                                        #  report_to="wandb",
                                         )

    print('====Initializing trainer====')
    trainer = Seq2SeqTrainer(model=model,
                             args=eval_args,
                             train_dataset=None,
                             eval_dataset=None,
                             data_collator=None,
                             tokenizer=tokenizer,
                             compute_metrics=None,
                             preprocess_logits_for_metrics=None
                             )
 
    # Process the conversation through the GOT pipeline
    
    # processed_input = DialoconanDatasetWithGraph(processed_input, "dev", tokenizer, 1024, 1024, None)
    # print(processed_input)
    results = trainer.predict(test_dataset=dataset, max_length=max_length)
    model_outputs, target_outputs = results.predictions, results.label_ids
    model_outputs = np.where(model_outputs != -100, model_outputs, tokenizer.pad_token_id)
    decoded_outputs = tokenizer.batch_decode(model_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_targets = tokenizer.batch_decode(target_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    cleaned_outputs = [output.strip() for output in decoded_outputs]
    generated_outputs = cleaned_outputs
    # print("__________________________", generated_outputs)
    return generated_outputs[0]

def main():
    st.title("Counterspeech Generation System")
    
    # Model selection from trained models
    model_dir = "experiments/DIALOCONAN"
    available_models = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    
    if not available_models:
        st.error("No trained models found in experiments/DIALOCONAN directory")
        return
        
    selected_model = st.sidebar.selectbox(
        "Select trained model", 
        available_models,
        index=0
    )
    
    model_path = os.path.join(model_dir, selected_model)
    model, tokenizer = load_model(model_path)
    
    # Initialize chat history and feedback
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        role_display = "HS" if message["role"] == "user" else "CN"
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add feedback buttons for bot responses
            if message["role"] == "assistant":
                message_id = str(i)
                col1, col2 = st.columns([1, 20])
                
                # Check if feedback was already given
                if message_id in st.session_state.feedback:
                    if st.session_state.feedback[message_id] == "thumbs_up":
                        col1.button("👍", key=f"up_{message_id}", disabled=True)
                        col2.button("👎", key=f"down_{message_id}", disabled=True)
                    else:
                        col1.button("👍", key=f"up_{message_id}", disabled=True)
                        col2.button("👎", key=f"down_{message_id}", disabled=True)
                else:
                    # No feedback yet, show active buttons
                    if col1.button("👍", key=f"up_{message_id}"):
                        st.session_state.feedback[message_id] = "thumbs_up"
                        # st.experimental_rerun()
                    if col2.button("👎", key=f"down_{message_id}"):
                        st.session_state.feedback[message_id] = "thumbs_down"
                        # st.experimental_rerun()
    
    # Input for new message
    if prompt := st.chat_input("Enter your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare full conversation context in dialoconan.csv format
        # Generate a unique dialogue_id if it doesn't exist
        if "dialogue_id" not in st.session_state:
            st.session_state.dialogue_id = str(uuid.uuid4())
            
        conversation_formatted = []
        for i, msg in enumerate(st.session_state.messages):
            turn_id = i + 1
            msg_type = "HS" if msg["role"] == "user" else "CN"
            entry = {
                "turn_id": turn_id,
                "text": msg["content"],
                "dialogue_id": st.session_state.dialogue_id,
                "type": msg_type
            }
            conversation_formatted.append(entry)
        
        # Show a spinner while generating response
        with st.spinner("Generating response..."):
            response = generate_response(model_path, tokenizer, conversation_formatted, max_length=256)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
            
            # Add feedback buttons for the new response
            message_id = str(len(st.session_state.messages) - 1)
            col1, col2 = st.columns([1, 20])
            if col1.button("👍", key=f"up_{message_id}"):
                st.session_state.feedback[message_id] = "thumbs_up"
                # st.experimental_rerun()
            if col2.button("👎", key=f"down_{message_id}"):
                st.session_state.feedback[message_id] = "thumbs_down"
                # st.experimental_rerun()
    
    # Add sidebar section to display feedback statistics
    if st.session_state.feedback:
        st.sidebar.markdown("### Feedback Statistics")
        thumbs_up = sum(1 for v in st.session_state.feedback.values() if v == "thumbs_up")
        thumbs_down = sum(1 for v in st.session_state.feedback.values() if v == "thumbs_down")
        st.sidebar.write(f"👍 Positive feedback: {thumbs_up}")
        st.sidebar.write(f"👎 Negative feedback: {thumbs_down}")
    
    # Add clear conversation button
    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.feedback = {}
        # st.experimental_rerun()

if __name__ == "__main__":
    main() 
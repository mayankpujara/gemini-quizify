import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))
from pdf_processing import DocumentProcessor
from embedding_client import EmbeddingClient
from chroma_collection_creator import ChromaCollectionCreator
from quiz_generator import QuizGenerator

class QuizManager:
    def __init__(self, questions: list):
        
        self.questions = questions
        self.total_questions = len(questions)

    def get_question_at_index(self, index: int):
        """
        Retrieves the quiz question object at the specified index. If the index is out of bounds,
        it restarts from the beginning index.
        """

        valid_index = index % self.total_questions
        return self.questions[valid_index]

    def next_question_index(self, direction=1):
        
        if st.session_state["question_index"] is not None:
            current_index = st.session_state["question_index"]
            new_index = (current_index + direction) % self.total_questions
            st.session_state["question_index"] = new_index
        else:
            raise ValueError("Session state not found")

if __name__ == "__main__":
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "radicalx-quizzify",
        "location": "us-central1"
    }
    
    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()
    
        embed_client = EmbeddingClient(**embed_config) 
    
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
    
        question = None
        question_bank = None
    
        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                
                st.write(topic_input)
                
                # Test the Quiz Generator
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question_bank = generator.generate_quiz()

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            
            quiz_manager = QuizManager(question_bank)
            with st.form("Multiple Choice Question"):

                index_question = quiz_manager.get_question_at_index(0)
                
                # Unpack choices for radio
                choices = []
                for choice in index_question['choices']: # For loop unpack the data structure
                    key = choice['key']
                    value = choice['value']
                    choices.append(f"{key}) {value}")
                

                # Display the question onto streamlit
                st.write(f"{index_question['question']}")
                
                answer = st.radio( # Display the radio button with the choices
                    'Choose the correct answer',
                    choices
                )
                st.form_submit_button("Submit")
                
                if submitted: # On click submit 
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key): # Check if answer is correct
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")

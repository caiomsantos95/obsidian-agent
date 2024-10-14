import sys
import subprocess
import pkg_resources
import os
import pickle
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import scrolledtext, ttk
from tkinter.font import Font
import threading
import shutil
import numpy as np
import re  # Import for regular expressions
import time  # Added import for time module
import logging
import yaml
from langchain_community.document_loaders import ObsidianLoader
from langchain.schema import Document
from config import OPENAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LangChain imports
from langchain.chains import ConversationChain, LLMChain
from langchain.agents import Tool, AgentExecutor, create_react_agent, AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import StringPromptTemplate, PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AgentAction, AgentFinish
from langchain_community.document_loaders import ObsidianLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Update this import
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Type hinting imports
from typing import List, Union, Dict, Any

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-4o-mini")  # You can choose the model you prefer

class CustomObsidianLoader(ObsidianLoader):
    def _parse_front_matter(self, content):
        # Check if the content starts with '---'
        if content.startswith('---'):
            # Find the end of the front matter
            end = content.find('---', 3)
            if end != -1:
                front_matter_text = content[3:end].strip()
                try:
                    return yaml.safe_load(front_matter_text)
                except yaml.YAMLError:
                    # If YAML parsing fails, return an empty dict
                    return {}
        # If no valid front matter is found, return an empty dict
        return {}

    def _get_metadata(self, file_path, content):
        metadata = super()._get_metadata(file_path, content)
        # Extract links from content
        links = re.findall(r'\[\[([^\]]+)\]\]', content)
        metadata['links'] = links
        return metadata

def load_or_create_vectorstore(obsidian_path, cache_dir="./cache", force_recreate=True):
    index_file = os.path.join(cache_dir, "faiss_index.pkl")
    metadata_file = os.path.join(cache_dir, "vectorstore_metadata.pkl")

    # If force_recreate is True, delete the existing cache
    if force_recreate and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logging.info("Deleted existing vectorstore cache.")

    os.makedirs(cache_dir, exist_ok=True)

    if not force_recreate and os.path.exists(index_file) and os.path.exists(metadata_file):
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        if datetime.now() - metadata["created_at"] < timedelta(days=1):
            logging.info("Loading vectorstore from cache...")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(cache_dir, embeddings)
            # Load the note_name_to_docs mapping
            with open(os.path.join(cache_dir, 'note_name_to_docs.pkl'), 'rb') as f:
                note_name_to_docs = pickle.load(f)
            return vectorstore, note_name_to_docs

    logging.info("Creating new vectorstore...")
    loader = CustomObsidianLoader(obsidian_path, collect_metadata=True)
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents from Obsidian vault")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(texts)} text chunks")

    # Build note_name_to_docs mapping
    note_name_to_docs = {}
    for doc in texts:
        # Extract note name from source
        note_name = os.path.splitext(os.path.basename(doc.metadata['source']))[0]
        if note_name in note_name_to_docs:
            note_name_to_docs[note_name].append(doc)
        else:
            note_name_to_docs[note_name] = [doc]

    embeddings = OpenAIEmbeddings()
    
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save the index
    vectorstore.save_local(cache_dir)

    with open(metadata_file, "wb") as f:
        pickle.dump({"created_at": datetime.now()}, f)

    # Save the note_name_to_docs mapping
    with open(os.path.join(cache_dir, 'note_name_to_docs.pkl'), 'wb') as f:
        pickle.dump(note_name_to_docs, f)

    return vectorstore, note_name_to_docs

obsidian_path = "/Users/caiodossantos/Library/Mobile Documents/iCloud~md~obsidian/Documents/cms"
vectorstore, note_name_to_docs = load_or_create_vectorstore(obsidian_path, force_recreate=False)

def create_obsidian_search_tool(vectorstore, note_name_to_docs):
    def obsidian_search(query: str) -> str:
        # Initial similarity search
        initial_docs = vectorstore.similarity_search(query, k=5)

        # Use a list instead of a set
        docs_list = list(initial_docs)

        # For each initial doc, get its linked notes
        for doc in initial_docs:
            links = doc.metadata.get('links', [])
            for link in links:
                linked_docs = note_name_to_docs.get(link, [])
                docs_list.extend(linked_docs)

        # Remove duplicates based on page_content
        unique_docs = []
        seen_content = set()
        for doc in docs_list:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)

        # Prepare content for summarization
        content_to_summarize = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in unique_docs
        ])

        custom_prompt = PromptTemplate(
            template="""You are an assistant summarizing content from an Obsidian Vault, which contains both knowledge articles and personal journal entries. Your goal is to create engaging and detailed summaries that bring the content to life while maintaining clarity and simplicity in language.

**Guidelines:**

1. **Distinguish Content Types:**
   - Clearly differentiate between **Knowledge Entries** and **Journal Entries**.
   - For **Knowledge Entries**, focus on explaining concepts, theories, and factual information.
   - For **Journal Entries**, highlight personal reflections, experiences, and anecdotes.

2. **Enhance Engagement:**
   - Use a narrative style that weaves together different pieces of information.
   - Incorporate interesting details and insights that make the summary compelling.
   - Avoid overly terse or bullet-point summaries; aim for flowing paragraphs.

3. **Provide Context and Depth:**
   - Offer background information where necessary to help understand the content.
   - Make connections between different pieces of information to provide a comprehensive view.

4. **Cite Sources:**
   - Always include the source of each piece of information in the format [Source: filename].
   - Integrate citations smoothly within the narrative.

5. **Maintain Clarity:**
   - Keep the language simple and natural.
   - Ensure the summary is well-organized and easy to follow.
6. **Context on Obsidian Vault:**
   - Obsidian used [[]] in the text to link to other notes.
   - If you find a link, you should explore that note and use the content from that note as context.

**Content to Summarize:**

{text}

**Please provide a detailed and engaging response that captures the key points, includes relevant sources, and highlights the most interesting aspects of the content.**""",
            input_variables=["text"]
        )

        summary_chain = LLMChain(llm=llm, prompt=custom_prompt)
        summary = summary_chain.run(content_to_summarize)

        return f"Summary of relevant information from your Obsidian Vault:\n{summary}"

    return Tool(
        name="Obsidian Search",
        func=obsidian_search,
        description="Search through Obsidian files for relevant information"
    )

def create_journal_tool(llm):
    def create_journal_entry(_: str) -> str:
        entry = f"# Journal Entry - {datetime.now().strftime('%Y-%m-%d')}\n\n"

        # Helper function to get user input
        def get_user_input(prompt):
            return input(f"LLM: {prompt}\nYou: ").strip()

        # Weekly questions
        entry += "## Weekly questions\n"
        weekly_questions = llm.predict("Generate 3 thoughtful weekly reflection questions.").split('\n')
        for question in weekly_questions:
            response = get_user_input(question)
            entry += f"- ### {question}\n{response}\n\n"

        # Plan for the week
        entry += "## Plan for the week\n"
        plan_prompt = llm.predict("Ask the user about their main goals or priorities for the upcoming week.")
        priorities = get_user_input(plan_prompt)
        entry += f"Priorities:\n{priorities}\n\n"

        # Journal entries
        entry += "## Journal entries\n"
        journal_prompt = llm.predict("Generate a prompt to encourage the user to reflect on their personal and professional life this week.")
        journal_entry = get_user_input(journal_prompt)

        # Ask for clarification or additional details
        clarification = llm.predict(f"Based on the user's response: '{journal_entry}', ask a follow-up question to encourage deeper reflection or clarification.")
        additional_thoughts = get_user_input(clarification)

        entry += f"{journal_entry}\n\nAdditional thoughts:\n{additional_thoughts}\n\n"

        # Quote of the week
        quote_prompt = llm.predict("Ask the user if they have a meaningful quote to share this week.")
        quote = get_user_input(quote_prompt)
        if quote.lower() not in ["no", "none", "n/a"]:
            entry += f"## Quote of the week\n{quote}\n\n"

        # Interesting articles/news/knowledge
        interesting_prompt = llm.predict("Ask the user about any interesting articles, news, or knowledge they came across this week.")
        interesting = get_user_input(interesting_prompt)
        if interesting.lower() not in ["no", "none", "n/a"]:
            entry += f"## Interesting articles/news/knowledge\n{interesting}\n\n"

        entry += "#journal"

        return entry

    return Tool(
        name="Journal",
        func=create_journal_entry,
        description="Create a structured journal entry based on user responses to LLM-generated questions"
    )

def create_gpt_tool(llm):
    def gpt_query(query: str) -> str:
        response = llm.predict(query)
        return f"GPT's response: {response}"

    return Tool(
        name="GPT Direct",
        func=gpt_query,
        description="Direct access to GPT model for general queries and tasks"
    )

class ChatbotGUI:
    def __init__(self, master, agent):
        self.master = master
        master.title("Obsidian Agent Chatbot")
        master.geometry("800x600")  # Increased size for better readability

        self.agent = agent

        self.chat_display = scrolledtext.ScrolledText(master, wrap=tk.WORD, bg="white")
        self.chat_display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Create tags for different colors and styles
        self.chat_display.tag_config("user", foreground="blue", font=("TkDefaultFont", 10, "bold"))
        self.chat_display.tag_config("agent", foreground="black", font=("TkDefaultFont", 10, "bold"))
        self.chat_display.tag_config("thinking", foreground="black", font=("TkDefaultFont", 10, "italic"))
        self.chat_display.tag_config("observation", foreground="black", font=("TkDefaultFont", 10, "italic"))
        self.chat_display.tag_config("thought", foreground="black", font=("TkDefaultFont", 10, "italic"))

        # Display initial message
        self.display_message("Agent: How can I help you?", "agent")

        self.button_frame = tk.Frame(master)
        self.button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.obsidian_button = tk.Button(self.button_frame, text="Obsidian Search", command=lambda: self.use_tool("Obsidian Search"))
        self.obsidian_button.pack(side=tk.LEFT, padx=5)

        self.journal_button = tk.Button(self.button_frame, text="Journal", command=lambda: self.use_tool("Journal"))
        self.journal_button.pack(side=tk.LEFT, padx=5)

        self.gpt_button = tk.Button(self.button_frame, text="GPT Direct", command=lambda: self.use_tool("GPT Direct"))
        self.gpt_button.pack(side=tk.LEFT, padx=5)

        self.input_frame = tk.Frame(master)
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.input_field = tk.Entry(self.input_frame)
        self.input_field.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.input_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

    def send_message(self, event=None):
        user_input = self.input_field.get()
        self.input_field.delete(0, tk.END)
        self.display_message(f"You: {user_input}", "user")

        threading.Thread(target=self.get_agent_response, args=(user_input,)).start()

    def get_agent_response(self, user_input):
        try:
            self.display_message("Agent is thinking...", "thinking")
            response = self.agent({"input": user_input})

            if isinstance(response, dict):
                if 'intermediate_steps' in response:
                    for i, step in enumerate(response['intermediate_steps']):
                        tool = step[0].tool
                        action = step[0].tool_input
                        result = step[1]

                        self.display_message(f"Step {i+1}: Using {tool}", "thought")
                        self.display_message(f"Action: {action}", "thought")
                        self.display_message(f"Observation: {result}", "observation")

                if 'output' in response:
                    self.display_message(f"Final Answer: {response['output']}", "agent")
                elif 'answer' in response:
                    self.display_message(f"Final Answer: {response['answer']}", "agent")
                else:
                    self.display_message(f"Final Answer: {response}", "agent")
            else:
                self.display_message(f"Final Answer: {response}", "agent")
        except Exception as e:
            self.display_message(f"Error: {str(e)}")
            logging.error(f"Detailed error: {e}")  # This will print the full error to the console

    def display_message(self, message, tag=None):
        if tag in ["user", "agent"]:
            self.chat_display.insert(tk.END, message + "\n\n", (tag, "bold"))
        elif tag in ["thinking", "observation", "thought"]:
            self.chat_display.insert(tk.END, message + "\n\n", (tag, "italic"))
        else:
            self.chat_display.insert(tk.END, message + "\n\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.update()

    def use_tool(self, tool_name):
        self.input_field.insert(tk.END, f"Use the {tool_name} tool to ")
        self.input_field.focus()

def main():
    try:
        search_tool = create_obsidian_search_tool(vectorstore, note_name_to_docs)
        journal_tool = create_journal_tool(llm)
        gpt_tool = create_gpt_tool(llm)

        tools = [search_tool, journal_tool, gpt_tool]

        memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            return_intermediate_steps=True
        )

        root = tk.Tk()
        gui = ChatbotGUI(root, agent)

        root.mainloop()

    except Exception as e:
        logging.error(f"An error occurred during setup: {e}")
        logging.error("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
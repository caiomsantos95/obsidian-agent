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
from config import OPENAI_API_KEY, OBSIDIAN_PATH
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from prompts import CUSTOM_SUMMARY_PROMPT
import tiktoken
import webbrowser
import urllib.parse

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# LangChain imports
from langchain.chains import LLMChain
from langchain.agents import Tool, create_react_agent, initialize_agent, AgentExecutor, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import StringPromptTemplate, PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AgentAction, AgentFinish
from langchain.document_loaders import ObsidianLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

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
        
        # Log the raw content for debugging
        logging.debug(f"Processing content from {os.path.basename(file_path)}:")
        logging.debug(f"Raw content (first 500 chars): {content[:500]}")
        
        # Extract different types of Obsidian links with more precise patterns
        wikilinks = re.findall(r'\[\[([^\]\|]+?)\]\]', content)  # [[Note]] - captures until ]] or |
        aliased_wikilinks = re.findall(r'\[\[([^\]\|]+?)\|([^\]]+?)\]\]', content)  # [[Note|Alias]]
        mdlinks = re.findall(r'\[([^\]]+)\]\(([^\)]+?\.md)\)', content)  # [Note](Note.md)
        bare_links = re.findall(r'(?<!\[)\b([a-zA-Z0-9-_\s]+\.md)\b', content)  # Note.md
        hashtags = re.findall(r'(?<![\w])#([a-zA-Z0-9-_]+)', content)  # #tag (not part of a word)
        
        # Combine all links
        all_links = set()
        
        # Process wiki-style links
        for link in wikilinks:
            # Remove any # and text after it (header links)
            base_link = link.split('#')[0].strip()
            if base_link:
                all_links.add(base_link)
                logging.debug(f"Added wiki link: {base_link}")
        
        # Process aliased wiki links
        for link, alias in aliased_wikilinks:
            base_link = link.split('#')[0].strip()
            if base_link:
                all_links.add(base_link)
                logging.debug(f"Added aliased wiki link: {base_link} (alias: {alias})")
        
        # Process markdown links
        for text, link in mdlinks:
            base_link = os.path.splitext(os.path.basename(link))[0].strip()
            if base_link:
                all_links.add(base_link)
                logging.debug(f"Added markdown link: {base_link}")
        
        # Process bare links
        for link in bare_links:
            base_link = os.path.splitext(link)[0].strip()
            if base_link:
                all_links.add(base_link)
                logging.debug(f"Added bare link: {base_link}")
        
        # Add hashtags as potential links
        all_links.update(hashtags)
        for tag in hashtags:
            logging.debug(f"Added hashtag: {tag}")
        
        logging.info(f"Extracted links from {os.path.basename(file_path)}:")
        logging.info(f"  Wiki-style links: {wikilinks}")
        logging.info(f"  Aliased wiki links: {[f'{link}|{alias}' for link, alias in aliased_wikilinks]}")
        logging.info(f"  Markdown links: {mdlinks}")
        logging.info(f"  Bare links: {bare_links}")
        logging.info(f"  Hashtags: {hashtags}")
        logging.info(f"  Final processed links: {list(all_links)}")
        
        metadata['links'] = list(all_links)
        return metadata

def load_or_create_vectorstore(obsidian_path, cache_dir="./cache", force_recreate=False):
    index_file = os.path.join(cache_dir, "faiss_index.pkl")
    metadata_file = os.path.join(cache_dir, "vectorstore_metadata.pkl")
    vault_state_file = os.path.join(cache_dir, "vault_state.pkl")

    os.makedirs(cache_dir, exist_ok=True)

    def get_vault_state(path):
        """Get the current state of the vault (modification times of all files)"""
        vault_state = {}
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.md'):
                    full_path = os.path.join(root, file)
                    vault_state[full_path] = os.path.getmtime(full_path)
        return vault_state

    def vault_has_changed():
        """Check if the vault has changed since last cache"""
        if not os.path.exists(vault_state_file):
            return True
        
        with open(vault_state_file, 'rb') as f:
            old_state = pickle.load(f)
        
        current_state = get_vault_state(obsidian_path)
        return old_state != current_state

    # Check if cache exists and is valid
    cache_exists = all(os.path.exists(f) for f in [index_file, metadata_file, vault_state_file])
    
    if not force_recreate and cache_exists and not vault_has_changed():
        try:
            logging.info("Loading vectorstore from cache...")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(cache_dir, embeddings)
            with open(os.path.join(cache_dir, 'note_name_to_docs.pkl'), 'rb') as f:
                note_name_to_docs = pickle.load(f)
            return vectorstore, note_name_to_docs
        except Exception as e:
            logging.warning(f"Error loading cache: {e}. Recreating vectorstore...")

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
        note_name = os.path.splitext(os.path.basename(doc.metadata['source']))[0]
        if note_name in note_name_to_docs:
            note_name_to_docs[note_name].append(doc)
        else:
            note_name_to_docs[note_name] = [doc]

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save everything
    vectorstore.save_local(cache_dir)
    with open(metadata_file, "wb") as f:
        pickle.dump({"created_at": datetime.now()}, f)
    with open(os.path.join(cache_dir, 'note_name_to_docs.pkl'), 'wb') as f:
        pickle.dump(note_name_to_docs, f)
    
    # Save current vault state
    with open(vault_state_file, 'wb') as f:
        pickle.dump(get_vault_state(obsidian_path), f)

    return vectorstore, note_name_to_docs

# Replace the hardcoded path with the imported variable
vectorstore, note_name_to_docs = load_or_create_vectorstore(OBSIDIAN_PATH, force_recreate=False)

def obsidian_search(query: str, vectorstore=vectorstore, note_name_to_docs=note_name_to_docs):
    logging.info(f"\nSearching for: {query}")
    
    # Initial search
    docs = vectorstore.similarity_search(query)
    logging.info(f"Initial search found {len(docs)} documents")
    
    all_docs = []  # Start with empty list for all documents
    explored_links = set()  # Track explored links
    valuable_links = set()  # Track valuable links
    
    # Process initial search results
    for doc in docs:
        logging.info(f"\nProcessing document: {doc.metadata.get('source', '').split('/')[-1]}")
        logging.info(f"Raw metadata: {doc.metadata}")
        
        # Add initial document with its content and source
        all_docs.append({
            'content': doc.page_content,
            'source': doc.metadata.get('source', '').split('/')[-1],
            'context': 'Main Search Result'
        })
        
        # Extract links using regex
        links = re.findall(r'\[\[(.*?)\]\]', doc.page_content)
        logging.info(f"Extracted links: {links}")
        
        # Explore each link
        for link in links:
            if link not in explored_links:
                explored_links.add(link)
                
                # Search for the linked document
                linked_docs = vectorstore.similarity_search(f"filename:{link}")
                
                if linked_docs:
                    valuable_links.add(link)
                    # Add linked document with context about why it was included
                    all_docs.append({
                        'content': linked_docs[0].page_content,
                        'source': linked_docs[0].metadata.get('source', '').split('/')[-1],
                        'context': f'Linked from {doc.metadata.get("source", "").split("/")[-1]}',
                        'link_reason': f'Referenced in discussion of {query}'
                    })

    logging.info("\nSummary of link exploration:")
    logging.info(f"Total explored links: {len(explored_links)}")
    logging.info(f"Total valuable links: {len(valuable_links)}")
    logging.info(f"Total documents for summarization: {len(all_docs)}")
    
    # Prepare content for summarization
    content_sections = []
    for doc in all_docs:
        section = f"[Source: {doc['source']}]\n"
        section += f"[Context: {doc['context']}]"
        if 'link_reason' in doc:
            section += f"\n[Link Reason: {doc['link_reason']}]"
        section += f"\n{doc['content']}\n"
        content_sections.append(section)
    
    content_to_summarize = "\n\n".join(content_sections)
    logging.info(f"Number of content sections for summarization: {len(content_sections)}")
    
    # Generate summary using the custom prompt
    summary_chain = LLMChain(llm=llm, prompt=CUSTOM_SUMMARY_PROMPT)
    summary = summary_chain.run(text=content_to_summarize)
    
    return summary

def create_obsidian_search_tool(vectorstore, note_name_to_docs):
    return Tool(
        name="Obsidian Search",
        func=obsidian_search,
        description="Search through Obsidian files for relevant information, evaluating and including content from linked notes"
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
        master.geometry("800x600")

        style = ttk.Style("cosmo")
        style.configure("TFrame", borderwidth=0)
        style.configure("Custom.TFrame", borderwidth=1, relief="solid", background="white")

        self.agent = agent

        self.chat_frame = ttk.Frame(master, padding=10)
        self.chat_frame.pack(expand=True, fill=tk.BOTH)

        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, bg="white")
        self.chat_display.pack(expand=True, fill=tk.BOTH)

        # Create tags for different colors and styles
        self.chat_display.tag_config("user", foreground="#007bff", font=("TkDefaultFont", 10, "bold"))
        self.chat_display.tag_config("agent", foreground="#28a745", font=("TkDefaultFont", 10, "bold"))
        self.chat_display.tag_config("thinking", foreground="#6c757d", font=("TkDefaultFont", 10, "italic"))
        self.chat_display.tag_config("observation", foreground="#17a2b8", font=("TkDefaultFont", 10))
        self.chat_display.tag_config("thought", foreground="#ffc107", font=("TkDefaultFont", 10))

        # Display initial message
        self.display_message("Agent: How can I help you?", "agent")

        self.button_frame = ttk.Frame(master, style="Custom.TFrame", padding=5)
        self.button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.obsidian_button = ttk.Button(self.button_frame, text="Obsidian Search", command=lambda: self.use_tool("Obsidian Search"))
        self.obsidian_button.pack(side=tk.LEFT, padx=5)

        self.journal_button = ttk.Button(self.button_frame, text="Journal", command=lambda: self.use_tool("Journal"))
        self.journal_button.pack(side=tk.LEFT, padx=5)

        self.gpt_button = ttk.Button(self.button_frame, text="GPT Direct", command=lambda: self.use_tool("GPT Direct"))
        self.gpt_button.pack(side=tk.LEFT, padx=5)

        self.input_frame = ttk.Frame(master, style="Custom.TFrame", padding=5)
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.input_field = ttk.Entry(self.input_frame)
        self.input_field.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.input_field.bind("<Return>", self.send_message)

        self.send_button = ttk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

        # Add a progress indicator
        self.progress_frame = ttk.Frame(self.input_frame)
        self.progress_frame.pack(side=tk.LEFT, padx=(5, 0))
        self.progress = ttk.Progressbar(self.progress_frame, mode="indeterminate", length=20)

    def send_message(self, event=None):
        user_input = self.input_field.get()
        self.input_field.delete(0, tk.END)
        self.display_message(f"You: {user_input}", "user")

        threading.Thread(target=self.get_agent_response, args=(user_input,)).start()

    def get_agent_response(self, user_input):
        try:
            self.display_message("Agent is thinking...", "thinking")
            self.progress.start()
            self.progress.pack()
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
            logging.error(f"Detailed error: {e}")
        finally:
            self.progress.stop()
            self.progress.pack_forget()

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

        root = ttk.Window(themename="cosmo")
        gui = ChatbotGUI(root, agent)

        root.mainloop()

    except Exception as e:
        logging.error(f"An error occurred during setup: {e}")
        logging.error("Please check your configuration and try again.")

if __name__ == "__main__":
    main()

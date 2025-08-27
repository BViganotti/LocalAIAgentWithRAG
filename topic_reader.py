import os
import PyPDF2
from typing import Dict, List, Optional, Tuple
import hashlib


class TopicReader:
    """Reads and processes topic files from the topic/ directory"""
    
    def __init__(self, topic_dir: str = "topic"):
        self.topic_dir = topic_dir
        self.supported_extensions = ['.pdf', '.txt', '.md']
    
    def get_topic_files(self) -> List[str]:
        """Get all supported topic files from the topic directory"""
        if not os.path.exists(self.topic_dir):
            os.makedirs(self.topic_dir)
            print(f"Created topic directory: {self.topic_dir}")
            return []
        
        topic_files = []
        for filename in os.listdir(self.topic_dir):
            file_path = os.path.join(self.topic_dir, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename)
                if ext.lower() in self.supported_extensions:
                    topic_files.append(file_path)
        
        return topic_files
    
    def read_txt_file(self, file_path: str) -> str:
        """Read content from a TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT file {file_path}: {e}")
            return ""
    
    def read_md_file(self, file_path: str) -> str:
        """Read content from a Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading MD file {file_path}: {e}")
            return ""
    
    def read_pdf_file(self, file_path: str) -> str:
        """Read content from a PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
            return ""
    
    def read_topic_file(self, file_path: str) -> Dict[str, str]:
        """Read a topic file and return its content with metadata"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Topic file not found: {file_path}")
        
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        
        # Read content based on file extension
        if ext.lower() == '.pdf':
            content = self.read_pdf_file(file_path)
        elif ext.lower() == '.txt':
            content = self.read_txt_file(file_path)
        elif ext.lower() == '.md':
            content = self.read_md_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Extract title from content (first line) or use filename
        title = name.replace('_', ' ').replace('-', ' ').title()
        if content.strip():
            lines = content.strip().split('\n')
            first_line = lines[0].strip()
            # If first line looks like a title (not too long, has title-like markers)
            if len(first_line) < 100 and any(marker in first_line for marker in ['#', 'Title:', 'Topic:']):
                title = first_line.strip('#').strip().strip(':').strip()
                content = '\n'.join(lines[1:]).strip()
        
        return {
            'file_path': file_path,
            'filename': filename,
            'title': title,
            'content': content,
            'file_type': ext.lower().replace('.', ''),
            'content_hash': hashlib.md5(content.encode()).hexdigest(),
            'word_count': len(content.split()) if content else 0,
            'char_count': len(content) if content else 0
        }
    
    def get_all_topics(self) -> List[Dict[str, str]]:
        """Get all topic files with their content and metadata"""
        topic_files = self.get_topic_files()
        topics = []
        
        for file_path in topic_files:
            try:
                topic_data = self.read_topic_file(file_path)
                topics.append(topic_data)
                print(f"Loaded topic: {topic_data['title']} ({topic_data['word_count']} words)")
            except Exception as e:
                print(f"Error loading topic from {file_path}: {e}")
        
        print(f"Total topics loaded: {len(topics)}")
        return topics
    
    def find_topic_by_title(self, title: str) -> Optional[Dict[str, str]]:
        """Find a specific topic by title (case-insensitive partial match)"""
        topics = self.get_all_topics()
        title_lower = title.lower()
        
        for topic in topics:
            if title_lower in topic['title'].lower():
                return topic
        
        return None
    
    def get_topic_list(self) -> List[Dict[str, str]]:
        """Get a simplified list of available topics (title and filename only)"""
        topics = self.get_all_topics()
        return [{'title': t['title'], 'filename': t['filename'], 'file_type': t['file_type']} 
                for t in topics]
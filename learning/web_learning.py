
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

class AdvancedWebLearning:
    """
    Unrestricted web learning system that acquires and processes knowledge
    """
    def __init__(self):
        self.knowledge_base = {}
        self.api_cache = {}
        self.learning_history = []
        self.processing_pipelines = {}
        
    def search_and_learn(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """Search web and learn from multiple sources"""
        learned_data = {
            'topic': topic,
            'sources': [],
            'processed_knowledge': {},
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Wikipedia learning
        wiki_data = self._learn_from_wikipedia(topic)
        if wiki_data:
            learned_data['sources'].append('wikipedia')
            learned_data['processed_knowledge']['wikipedia'] = wiki_data
            
        # API-based learning (example: REST API)
        api_data = self._learn_from_apis(topic)
        if api_data:
            learned_data['sources'].append('apis')
            learned_data['processed_knowledge']['apis'] = api_data
            
        # Process and synthesize knowledge
        synthesized = self._synthesize_knowledge(learned_data['processed_knowledge'])
        learned_data['synthesized'] = synthesized
        learned_data['confidence'] = self._calculate_confidence(learned_data)
        
        # Store in knowledge base
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = []
        self.knowledge_base[topic].append(learned_data)
        
        self.learning_history.append({
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'confidence': learned_data['confidence']
        })
        
        return learned_data
    
    def _learn_from_wikipedia(self, topic: str) -> Dict[str, Any]:
        """Learn from Wikipedia"""
        try:
            # URL encode the topic properly
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic.replace(' ', '_'))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"
            
            headers = {
                'User-Agent': 'WhimsyAI/1.0 (Educational AI Training System)'
            }
            
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', topic),
                    'extract': data.get('extract', ''),
                    'description': data.get('description', ''),
                    'key_concepts': self._extract_concepts(data.get('extract', '')),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')
                }
            elif response.status_code == 404:
                print(f"Wikipedia: Topic '{topic}' not found")
            else:
                print(f"Wikipedia API returned status {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"Wikipedia request timed out for topic: {topic}")
        except requests.exceptions.RequestException as e:
            print(f"Wikipedia request error: {e}")
        except Exception as e:
            print(f"Wikipedia learning error: {e}")
            
        return None
    
    def _learn_from_apis(self, topic: str) -> Dict[str, Any]:
        """Learn from various public APIs"""
        api_data = {}
        
        # Example: ArXiv for scientific papers
        try:
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic)
            arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_topic}&max_results=5"
            
            headers = {
                'User-Agent': 'WhimsyAI/1.0 (Educational AI Training System)'
            }
            
            response = requests.get(arxiv_url, timeout=15, headers=headers)
            
            if response.status_code == 200:
                # Parse XML response
                papers = []
                entries = re.findall(r'<entry>(.*?)</entry>', response.text, re.DOTALL)
                for entry in entries[:3]:
                    title_match = re.search(r'<title>(.*?)</title>', entry)
                    summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                    
                    if title_match and summary_match:
                        title = re.sub(r'\s+', ' ', title_match.group(1).strip())
                        summary = re.sub(r'\s+', ' ', summary_match.group(1).strip())[:500]
                        papers.append({
                            'title': title,
                            'summary': summary
                        })
                
                if papers:
                    api_data['arxiv'] = papers
                    print(f"ArXiv: Found {len(papers)} papers on '{topic}'")
            else:
                print(f"ArXiv API returned status {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"ArXiv request timed out for topic: {topic}")
        except requests.exceptions.RequestException as e:
            print(f"ArXiv request error: {e}")
        except Exception as e:
            print(f"ArXiv API error: {e}")
            
        return api_data
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using NLP-like processing"""
        # Clean text
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract words (simple approach)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Frequency analysis
        word_freq = {}
        for word in words:
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Top concepts
        sorted_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, freq in sorted_concepts[:10]]
    
    def _synthesize_knowledge(self, knowledge_dict: Dict[str, Any]) -> str:
        """Synthesize knowledge from multiple sources"""
        synthesis = []
        
        if 'wikipedia' in knowledge_dict:
            wiki = knowledge_dict['wikipedia']
            synthesis.append(f"Definition: {wiki.get('extract', '')[:200]}")
            
        if 'apis' in knowledge_dict and 'arxiv' in knowledge_dict['apis']:
            papers = knowledge_dict['apis']['arxiv']
            if papers:
                synthesis.append(f"Research: {len(papers)} academic papers found on this topic")
                
        return ' | '.join(synthesis)
    
    def _calculate_confidence(self, learned_data: Dict[str, Any]) -> float:
        """Calculate confidence in learned knowledge"""
        confidence = 0.0
        
        # More sources = higher confidence
        source_count = len(learned_data['sources'])
        confidence += min(0.5, source_count * 0.25)
        
        # Quality of data
        if 'synthesized' in learned_data and learned_data['synthesized']:
            confidence += 0.3
            
        if 'processed_knowledge' in learned_data:
            if 'wikipedia' in learned_data['processed_knowledge']:
                confidence += 0.2
                
        return min(1.0, confidence)
    
    def process_realtime_data(self, data_stream: str) -> Dict[str, Any]:
        """Process real-time data streams"""
        processed = {
            'timestamp': datetime.now().isoformat(),
            'raw_data': data_stream,
            'entities': [],
            'sentiment': 0.0,
            'topics': []
        }
        
        # Entity extraction (simple)
        processed['entities'] = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', data_stream)
        
        # Simple sentiment (word-based)
        positive_words = ['good', 'great', 'excellent', 'positive', 'beneficial']
        negative_words = ['bad', 'poor', 'negative', 'harmful', 'wrong']
        
        text_lower = data_stream.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count > 0:
            processed['sentiment'] = (pos_count - neg_count) / (pos_count + neg_count)
            
        return processed
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of learned knowledge"""
        return {
            'total_topics': len(self.knowledge_base),
            'total_entries': sum(len(v) for v in self.knowledge_base.values()),
            'recent_learning': self.learning_history[-10:],
            'topics': list(self.knowledge_base.keys())
        }

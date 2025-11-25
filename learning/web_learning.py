
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import time

class AdvancedWebLearning:
    """
    Multi-source web learning system that acquires knowledge from anywhere
    Mimics human learning by pulling from diverse sources and synthesizing information
    """
    def __init__(self):
        self.knowledge_base = {}
        self.api_cache = {}
        self.learning_history = []
        self.processing_pipelines = {}
        self.source_reliability = {}  # Track source quality
        
        # API endpoints for diverse learning
        self.apis = {
            'wikipedia': 'https://en.wikipedia.org/api/rest_v1/page/summary/',
            'arxiv': 'http://export.arxiv.org/api/query',
            'github': 'https://api.github.com/search/repositories',
            'stackoverflow': 'https://api.stackexchange.com/2.3/search/advanced',
            'news': 'https://newsapi.org/v2/everything',
            'reddit': 'https://www.reddit.com/search.json',
        }
        
    def search_and_learn(self, topic: str, depth: int = 5) -> Dict[str, Any]:
        """Search web and learn from ALL available sources"""
        learned_data = {
            'topic': topic,
            'sources': [],
            'processed_knowledge': {},
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'cross_validation': []
        }
        
        # Learn from multiple sources in parallel
        sources_data = {}
        
        # Wikipedia - Encyclopedia knowledge
        wiki_data = self._learn_from_wikipedia(topic)
        if wiki_data:
            sources_data['wikipedia'] = wiki_data
            learned_data['sources'].append('wikipedia')
            
        # ArXiv - Academic research
        arxiv_data = self._learn_from_arxiv(topic)
        if arxiv_data:
            sources_data['arxiv'] = arxiv_data
            learned_data['sources'].append('arxiv')
            
        # GitHub - Code and documentation
        github_data = self._learn_from_github(topic)
        if github_data:
            sources_data['github'] = github_data
            learned_data['sources'].append('github')
            
        # Stack Overflow - Technical Q&A
        stackoverflow_data = self._learn_from_stackoverflow(topic)
        if stackoverflow_data:
            sources_data['stackoverflow'] = stackoverflow_data
            learned_data['sources'].append('stackoverflow')
            
        # Reddit - Community discussions and real-world applications
        reddit_data = self._learn_from_reddit(topic)
        if reddit_data:
            sources_data['reddit'] = reddit_data
            learned_data['sources'].append('reddit')
        
        # General web scraping - Learn from any accessible content
        web_data = self._learn_from_web_search(topic)
        if web_data:
            sources_data['web'] = web_data
            learned_data['sources'].append('web')
        
        learned_data['processed_knowledge'] = sources_data
        
        # Cross-validate information across sources
        learned_data['cross_validation'] = self._cross_validate_sources(sources_data, topic)
        
        # Synthesize knowledge with deep understanding
        synthesized = self._synthesize_knowledge_deep(sources_data, topic)
        learned_data['synthesized'] = synthesized
        
        # Calculate confidence based on source diversity and agreement
        learned_data['confidence'] = self._calculate_confidence_advanced(learned_data)
        
        # Store in knowledge base with metadata
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = []
        self.knowledge_base[topic].append(learned_data)
        
        self.learning_history.append({
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'confidence': learned_data['confidence'],
            'source_count': len(learned_data['sources'])
        })
        
        return learned_data
    
    def _learn_from_wikipedia(self, topic: str) -> Dict[str, Any]:
        """Learn from Wikipedia - Encyclopedia knowledge"""
        try:
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic.replace(' ', '_'))
            url = f"{self.apis['wikipedia']}{encoded_topic}"
            
            headers = {'User-Agent': 'WhimsyAI/2.0 (Advanced Learning System)'}
            response = requests.get(url, timeout=8, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', topic),
                    'extract': data.get('extract', ''),
                    'description': data.get('description', ''),
                    'key_concepts': self._extract_concepts(data.get('extract', '')),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'quality_score': 0.9  # Wikipedia is generally high quality
                }
        except Exception as e:
            print(f"Wikipedia learning error: {e}")
        return None
    
    def _learn_from_arxiv(self, topic: str) -> Dict[str, Any]:
        """Learn from ArXiv - Academic research"""
        try:
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic)
            url = f"{self.apis['arxiv']}?search_query=all:{encoded_topic}&max_results=5"
            
            headers = {'User-Agent': 'WhimsyAI/2.0 (Advanced Learning System)'}
            response = requests.get(url, timeout=12, headers=headers)
            
            if response.status_code == 200:
                papers = []
                entries = re.findall(r'<entry>(.*?)</entry>', response.text, re.DOTALL)
                for entry in entries[:3]:
                    title_match = re.search(r'<title>(.*?)</title>', entry)
                    summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                    
                    if title_match and summary_match:
                        title = re.sub(r'\s+', ' ', title_match.group(1).strip())
                        summary = re.sub(r'\s+', ' ', summary_match.group(1).strip())[:600]
                        papers.append({'title': title, 'summary': summary})
                
                if papers:
                    print(f"ArXiv: Found {len(papers)} papers on '{topic}'")
                    return {
                        'papers': papers,
                        'count': len(papers),
                        'quality_score': 0.95,  # Academic papers are very high quality
                        'insights': self._extract_research_insights(papers)
                    }
        except Exception as e:
            print(f"ArXiv error: {e}")
        return None
    
    def _learn_from_github(self, topic: str) -> Dict[str, Any]:
        """Learn from GitHub - Code repositories and documentation"""
        try:
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic)
            url = f"{self.apis['github']}?q={encoded_topic}&sort=stars&per_page=5"
            
            headers = {
                'User-Agent': 'WhimsyAI/2.0 (Advanced Learning System)',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                repos = []
                for item in data.get('items', [])[:5]:
                    repos.append({
                        'name': item.get('name'),
                        'description': item.get('description', ''),
                        'stars': item.get('stargazers_count', 0),
                        'language': item.get('language', ''),
                        'topics': item.get('topics', [])
                    })
                
                if repos:
                    print(f"GitHub: Found {len(repos)} repositories on '{topic}'")
                    return {
                        'repositories': repos,
                        'count': len(repos),
                        'quality_score': 0.85,
                        'languages': list(set([r['language'] for r in repos if r['language']])),
                        'common_topics': self._extract_common_topics(repos)
                    }
        except Exception as e:
            print(f"GitHub error: {e}")
        return None
    
    def _learn_from_stackoverflow(self, topic: str) -> Dict[str, Any]:
        """Learn from Stack Overflow - Technical Q&A"""
        try:
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic)
            url = f"{self.apis['stackoverflow']}?order=desc&sort=votes&q={encoded_topic}&site=stackoverflow"
            
            headers = {'User-Agent': 'WhimsyAI/2.0 (Advanced Learning System)'}
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                questions = []
                for item in data.get('items', [])[:5]:
                    questions.append({
                        'title': item.get('title'),
                        'score': item.get('score', 0),
                        'tags': item.get('tags', []),
                        'is_answered': item.get('is_answered', False),
                        'view_count': item.get('view_count', 0)
                    })
                
                if questions:
                    print(f"StackOverflow: Found {len(questions)} Q&A on '{topic}'")
                    return {
                        'questions': questions,
                        'count': len(questions),
                        'quality_score': 0.8,
                        'common_tags': self._extract_common_tags(questions),
                        'practical_applications': len([q for q in questions if q['is_answered']])
                    }
        except Exception as e:
            print(f"StackOverflow error: {e}")
        return None
    
    def _learn_from_reddit(self, topic: str) -> Dict[str, Any]:
        """Learn from Reddit - Community discussions"""
        try:
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic)
            url = f"{self.apis['reddit']}?q={encoded_topic}&limit=10&sort=relevance"
            
            headers = {'User-Agent': 'WhimsyAI/2.0 (Advanced Learning System)'}
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                posts = []
                for child in data.get('data', {}).get('children', [])[:5]:
                    post_data = child.get('data', {})
                    posts.append({
                        'title': post_data.get('title'),
                        'score': post_data.get('score', 0),
                        'subreddit': post_data.get('subreddit'),
                        'num_comments': post_data.get('num_comments', 0),
                        'selftext': post_data.get('selftext', '')[:200]
                    })
                
                if posts:
                    print(f"Reddit: Found {len(posts)} discussions on '{topic}'")
                    return {
                        'posts': posts,
                        'count': len(posts),
                        'quality_score': 0.6,  # Reddit quality varies
                        'community_interest': sum([p['score'] for p in posts]),
                        'active_communities': list(set([p['subreddit'] for p in posts]))
                    }
        except Exception as e:
            print(f"Reddit error: {e}")
        return None
    
    def _learn_from_web_search(self, topic: str) -> Dict[str, Any]:
        """Learn from general web content"""
        try:
            # Use DuckDuckGo HTML search (no API key needed)
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic)
            url = f"https://html.duckduckgo.com/html/?q={encoded_topic}"
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                
                for result in soup.find_all('div', class_='result__body')[:5]:
                    title_elem = result.find('a', class_='result__a')
                    snippet_elem = result.find('a', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'snippet': snippet_elem.get_text(strip=True),
                            'url': title_elem.get('href', '')
                        })
                
                if results:
                    print(f"Web: Found {len(results)} results on '{topic}'")
                    return {
                        'results': results,
                        'count': len(results),
                        'quality_score': 0.7,
                        'diverse_perspectives': True
                    }
        except Exception as e:
            print(f"Web search error: {e}")
        return None
    
    def _extract_research_insights(self, papers: List[Dict]) -> List[str]:
        """Extract key insights from research papers"""
        insights = []
        for paper in papers:
            summary = paper.get('summary', '')
            # Extract sentences with key research indicators
            sentences = re.split(r'[.!?]+', summary)
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['we propose', 'we show', 'our results', 'we demonstrate', 'novel', 'improve']):
                    insights.append(sentence.strip())
        return insights[:5]
    
    def _extract_common_topics(self, repos: List[Dict]) -> List[str]:
        """Extract common topics from GitHub repositories"""
        all_topics = []
        for repo in repos:
            all_topics.extend(repo.get('topics', []))
        
        topic_freq = {}
        for topic in all_topics:
            topic_freq[topic] = topic_freq.get(topic, 0) + 1
        
        return [topic for topic, count in sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _extract_common_tags(self, questions: List[Dict]) -> List[str]:
        """Extract common tags from Stack Overflow questions"""
        all_tags = []
        for q in questions:
            all_tags.extend(q.get('tags', []))
        
        tag_freq = {}
        for tag in all_tags:
            tag_freq[tag] = tag_freq.get(tag, 0) + 1
        
        return [tag for tag, count in sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        word_freq = {}
        for word in words:
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, freq in sorted_concepts[:10]]
    
    def _cross_validate_sources(self, sources_data: Dict[str, Any], topic: str) -> List[Dict]:
        """Cross-validate information across multiple sources"""
        validations = []
        
        # Check if multiple sources mention similar concepts
        all_concepts = []
        for source, data in sources_data.items():
            if source == 'wikipedia' and 'key_concepts' in data:
                all_concepts.extend(data['key_concepts'])
            elif source == 'github' and 'common_topics' in data:
                all_concepts.extend(data['common_topics'])
            elif source == 'stackoverflow' and 'common_tags' in data:
                all_concepts.extend(data['common_tags'])
        
        # Find concepts mentioned by multiple sources
        concept_freq = {}
        for concept in all_concepts:
            concept_freq[concept] = concept_freq.get(concept, 0) + 1
        
        validated_concepts = [c for c, freq in concept_freq.items() if freq >= 2]
        
        if validated_concepts:
            validations.append({
                'type': 'cross_source_validation',
                'validated_concepts': validated_concepts,
                'agreement_level': len(validated_concepts) / max(len(all_concepts), 1)
            })
        
        return validations
    
    def _synthesize_knowledge_deep(self, sources_data: Dict[str, Any], topic: str) -> str:
        """Synthesize knowledge from all sources with deep understanding"""
        synthesis_parts = []
        
        # Academic understanding
        if 'arxiv' in sources_data:
            arxiv = sources_data['arxiv']
            papers = arxiv.get('papers', [])
            insights = arxiv.get('insights', [])
            if papers:
                synthesis_parts.append(f"Academic Research: {len(papers)} papers analyzed")
            if insights:
                synthesis_parts.append(f"Key Findings: {insights[0][:150]}")
        
        # Conceptual understanding
        if 'wikipedia' in sources_data:
            wiki = sources_data['wikipedia']
            extract = wiki.get('extract', '')
            if extract:
                synthesis_parts.append(f"Core Concept: {extract[:200]}")
        
        # Practical implementation
        if 'github' in sources_data:
            github = sources_data['github']
            repos = github.get('repositories', [])
            languages = github.get('languages', [])
            if repos:
                synthesis_parts.append(f"Practical Implementations: {len(repos)} projects in {', '.join(languages[:3])}")
        
        # Technical knowledge
        if 'stackoverflow' in sources_data:
            so = sources_data['stackoverflow']
            questions = so.get('questions', [])
            if questions:
                answered = so.get('practical_applications', 0)
                synthesis_parts.append(f"Technical Q&A: {len(questions)} questions, {answered} solved")
        
        # Community perspective
        if 'reddit' in sources_data:
            reddit = sources_data['reddit']
            communities = reddit.get('active_communities', [])
            if communities:
                synthesis_parts.append(f"Community Engagement: Active in r/{', r/'.join(communities[:3])}")
        
        # Web insights
        if 'web' in sources_data:
            web = sources_data['web']
            if web.get('diverse_perspectives'):
                synthesis_parts.append(f"Diverse Web Perspectives: {web.get('count', 0)} sources analyzed")
        
        if not synthesis_parts:
            return f"Learning about {topic} from multiple sources..."
        
        return ' | '.join(synthesis_parts)
    
    def _calculate_confidence_advanced(self, learned_data: Dict[str, Any]) -> float:
        """Calculate confidence using advanced metrics"""
        confidence = 0.0
        
        # Source diversity (more diverse sources = higher confidence)
        source_count = len(learned_data['sources'])
        confidence += min(0.4, source_count * 0.08)
        
        # Cross-validation (agreement across sources)
        if learned_data.get('cross_validation'):
            for validation in learned_data['cross_validation']:
                if validation['type'] == 'cross_source_validation':
                    confidence += validation['agreement_level'] * 0.3
        
        # Quality of synthesis
        if learned_data.get('synthesized') and len(learned_data['synthesized']) > 100:
            confidence += 0.2
        
        # Source quality weighting
        quality_scores = []
        for source, data in learned_data.get('processed_knowledge', {}).items():
            if isinstance(data, dict) and 'quality_score' in data:
                quality_scores.append(data['quality_score'])
        
        if quality_scores:
            confidence += np.mean(quality_scores) * 0.1
        
        return min(1.0, confidence)
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get comprehensive knowledge summary"""
        return {
            'total_topics': len(self.knowledge_base),
            'total_entries': sum(len(v) for v in self.knowledge_base.values()),
            'recent_learning': self.learning_history[-20:],
            'topics': list(self.knowledge_base.keys()),
            'source_coverage': {
                'wikipedia': sum(1 for h in self.learning_history if 'wikipedia' in str(h)),
                'arxiv': sum(1 for h in self.learning_history if 'arxiv' in str(h)),
                'github': sum(1 for h in self.learning_history if 'github' in str(h)),
                'stackoverflow': sum(1 for h in self.learning_history if 'stackoverflow' in str(h)),
                'reddit': sum(1 for h in self.learning_history if 'reddit' in str(h)),
                'web': sum(1 for h in self.learning_history if 'web' in str(h))
            }
        }

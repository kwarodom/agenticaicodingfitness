# PYTHON — research_pipeline.py
import os
from dotenv import load_dotenv

load_dotenv()

import anthropic
import json
from datetime import datetime

client = anthropic.Anthropic()

class ResearchPipeline:
    def __init__(self):
        self.state = {
            "topic": "",
            "queries": [],
            "sources": [],
            "summaries": [],
            "report": "",
            "quality_score": 0,
            "log": []
        }
    
    def _log(self, step, msg):
        entry = {"step": step, "time": datetime.now().isoformat(), "msg": msg}
        self.state["log"].append(entry)
        print(f"  [{step}] {msg}")
    
    def _ask(self, prompt, max_tokens=1024):
        """Helper: single Claude call"""
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text
    
    def _parse_json(self, text):
        """Helper: Extract and parse JSON from Claude's response"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        # Fallback to finding the first { or [ if there's text before/after
        if not (text.startswith('{') or text.startswith('[')):
            start = text.find('{') if '{' in text else text.find('[')
            end = text.rfind('}') if '}' in text else text.rfind(']')
            if start != -1 and end != -1:
                text = text[start:end+1]
                
        return json.loads(text)
    
    def step1_generate_queries(self, topic):
        """Generate 3 search queries for the topic"""
        self.state["topic"] = topic
        self._log("QUERIES", f"Generating queries for: {topic}")
        
        result = self._ask(f"""Generate exactly 3 search queries to research: "{topic}"
Return as JSON array: ["query1", "query2", "query3"]
Only return the JSON, nothing else.""")
        
        self.state["queries"] = self._parse_json(result)
        self._log("QUERIES", f"Generated: {self.state['queries']}")
    
    def step2_search(self):
        """Web search using DuckDuckGo API"""
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            for q in self.state["queries"]:
                self._log("SEARCH", f"Searching: {q}")
                try:
                    results = list(ddgs.text(q, max_results=3))
                    if not results:
                        source = f"No results found for query: {q}"
                    else:
                        snippets = [f"Source: {res.get('title', '')}\nURL: {res.get('href', '')}\nExcerpt: {res.get('body', '')}" for res in results]
                        source = "\n\n".join(snippets)
                except Exception as e:
                    self._log("SEARCH ERROR", f"Error searching {q}: {e}")
                    source = f"Search failed for query: {q}. Error: {e}"
                
                self.state["sources"].append({"query": q, "content": source})
    
    def step3_summarize(self):
        """Summarize each source"""
        for src in self.state["sources"]:
            self._log("SUMMARIZE", f"Summarizing source for: {src['query']}")
            summary = self._ask(f"""Summarize this in 3 key bullet points:
{src['content']}
Format: - Key point 1\n- Key point 2\n- Key point 3""")
            self.state["summaries"].append(summary)
    
    def step4_synthesize(self):
        """Combine all summaries into final report"""
        self._log("SYNTHESIZE", "Creating final report...")
        all_summaries = "\n\n".join(
            [f"Source {i+1}:\n{s}" for i, s in enumerate(self.state["summaries"])]
        )
        self.state["report"] = self._ask(f"""You are a research analyst.
Synthesize these findings into a coherent 300-word report on "{self.state['topic']}":

{all_summaries}

Structure: Introduction → Key Findings → Implications → Conclusion""")
    
    def step5_quality_score(self):
        """Rate the report quality"""
        self._log("QA", "Scoring report quality...")
        score_result = self._ask(f"""Rate this research report 1-10 for:
- Accuracy (are claims supported?)
- Completeness (are key aspects covered?)
- Clarity (is it well-written?)
- Usefulness (would someone act on this?)

Report:
{self.state['report']}

Return as JSON: {{"accuracy": N, "completeness": N, "clarity": N, "usefulness": N, "overall": N, "feedback": "..."}}""")
        self.state["quality_score"] = self._parse_json(score_result)
    
    def run(self, topic):
        """Execute the full pipeline"""
        print(f"\n{'='*60}")
        print(f"RESEARCH PIPELINE: {topic}")
        print(f"{'='*60}\n")
        
        self.step1_generate_queries(topic)
        self.step2_search()
        self.step3_summarize()
        self.step4_synthesize()
        self.step5_quality_score()
        
        print(f"\n{'='*60}")
        print("FINAL REPORT:")
        print(f"{'='*60}")
        print(self.state["report"])
        print(f"\nQuality Score: {self.state['quality_score']}")
        
        # Save report to a file for external use
        with open("final_report.md", "w") as f:
            f.write(self.state["report"])
            
        return self.state

# RUN IT
pipeline = ResearchPipeline()
result = pipeline.run("AI-powered building energy optimization in Southeast Asia")
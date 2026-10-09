"""
Paper-to-Content Studio — Part 2 Feature #3
Convert papers into multiple formats: ELI5, slides, tweets, podcast scripts
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from core.ner_extractor import extract_technical_entities

class ContentFormat(Enum):
    """Supported output formats"""
    ELI5 = "eli5"  # Explain Like I'm 5
    SLIDE_OUTLINE = "slide_outline"
    TWEET_THREAD = "tweet_thread"
    PODCAST_SCRIPT = "podcast_script"
    INFOGRAPHIC_BRIEF = "infographic_brief"
    BLOG_POST = "blog_post"

@dataclass
class ContentOutput:
    """Generated content in a specific format"""
    format: str
    title: str
    content: str  # Main content (Markdown)
    metadata: Dict
    export_formats: Dict  # {format_name: formatted_content}

class PaperContentStudio:
    """Generate multiple content formats from papers"""
    
    def __init__(self, llm_router, rag_pipeline):
        self.llm = llm_router
        self.rag = rag_pipeline
    
    def generate_all_formats(
        self,
        paper_id: str,
        paper_data: Dict,
        target_audience: Optional[str] = None
    ) -> Dict[str, ContentOutput]:
        """Generate all content formats for a paper"""
        
        results = {}
        
        for format_type in ContentFormat:
            results[format_type.value] = self.generate_format(
                paper_id,
                paper_data,
                format_type,
                target_audience
            )
        
        return results
    
    def generate_format(
        self,
        paper_id: str,
        paper_data: Dict,
        format_type: ContentFormat,
        target_audience: Optional[str] = None
    ) -> ContentOutput:
        """Generate content in a specific format"""
        
        # Extract paper content
        title = paper_data.get("title", "Untitled")
        abstract = paper_data.get("abstract", "")
        
        # Retrieve key sections from RAG
        chunks = self.rag.retrieve(paper_id, top_k=5)
        main_content = "\n\n".join([c["content"] for c in chunks])
        
        if format_type == ContentFormat.ELI5:
            return self._generate_eli5(title, abstract, main_content)
        
        elif format_type == ContentFormat.SLIDE_OUTLINE:
            return self._generate_slide_outline(title, abstract, main_content)
        
        elif format_type == ContentFormat.TWEET_THREAD:
            return self._generate_tweet_thread(title, abstract, main_content)
        
        elif format_type == ContentFormat.PODCAST_SCRIPT:
            return self._generate_podcast_script(
                title, abstract, main_content, target_audience
            )
        
        elif format_type == ContentFormat.INFOGRAPHIC_BRIEF:
            return self._generate_infographic_brief(title, abstract, main_content)
        
        elif format_type == ContentFormat.BLOG_POST:
            return self._generate_blog_post(title, abstract, main_content, target_audience)
    
    def _generate_eli5(
        self,
        title: str,
        abstract: str,
        content: str
    ) -> ContentOutput:
        """Generate an ELI5 (Explain Like I'm 5) summary"""
        
        prompt = f"""
Create an "Explain Like I'm 5" summary of this research paper.
Use very simple language, analogies, and short sentences.
Avoid jargon. If you must use technical terms, explain them immediately.

Title: {title}

Abstract: {abstract}

Content: {content[:2000]}

Format your response as:
## What is this paper about?
[1-2 sentences]

## The Big Idea
[2-3 simple analogies]

## Why does it matter?
[Real-world applications in simple terms]

## The Simple Version
[3-4 bullet points of key findings]
"""
        
        response = self.llm.generate(prompt, temperature=0.7, max_tokens=800)
        
        return ContentOutput(
            format="eli5",
            title=f"ELI5: {title}",
            content=response,
            metadata={
                "target_audience": "General public",
                "reading_time_minutes": 3,
                "tone": "Conversational, simple"
            },
            export_formats={
                "markdown": response,
                "html": self._markdown_to_html(response),
                "text": response
            }
        )
    
    def _generate_slide_outline(
        self,
        title: str,
        abstract: str,
        content: str
    ) -> ContentOutput:
        """Generate slide deck outline using NER"""
        
        # 1. Extract technical entities using spaCy
        entities = extract_technical_entities(f"{title} {abstract} {content}")
        
        # 2. Format entities for the prompt
        entity_context = f"""
TECHNICAL ENTITIES TO FEATURE:
- Key Algorithms/Models: {', '.join(entities['algorithms']) or 'None extracted'}
- Datasets/Benchmarks: {', '.join(entities['datasets']) or 'None extracted'}
- Key Metrics: {', '.join(entities['metrics']) or 'None extracted'}
- Organizations/Authors: {', '.join(entities['organizations']) or 'None extracted'}
"""

        prompt = f"""
Create a slide deck outline for presenting this paper.
Generate 8-10 slides that tell the complete story from intro to conclusions.

CRITICAL INSTRUCTION: You MUST create a dedicated slide titled "Technologies & Datasets" that explicitly lists the items provided in the TECHNICAL ENTITIES section below.

Title: {title}
Abstract: {abstract}
Content: {content[:1500]}

{entity_context}

For each slide, provide:
- Slide number
- Title
- 3-4 bullet points
- Speaker notes (brief)

Format as JSON:
{{
  "slides": [
    {{"number": 1, "title": "...", "bullets": [...], "notes": "..."}}
  ]
}}
"""
        
        response = self.llm.generate(
            prompt, temperature=0.6, max_tokens=2000, format_json=True
        )
        
        try:
            slide_data = json.loads(response)
        except:
            slide_data = {"slides": []}
        
        # Convert to markdown outline
        markdown_outline = self._slides_to_markdown(slide_data.get("slides", []))
        
        return ContentOutput(
            format="slide_outline",
            title=f"Slides: {title}",
            content=markdown_outline,
            metadata={
                "num_slides": len(slide_data.get("slides", [])),
                "presentation_duration_minutes": 20,
                "format": "PPTX template"
            },
            export_formats={
                "markdown": markdown_outline,
                "json": json.dumps(slide_data, indent=2),
                "pptx": "[Can be generated with python-pptx]"
            }
        )
    
    def _generate_tweet_thread(
        self,
        title: str,
        abstract: str,
        content: str
    ) -> ContentOutput:
        """Generate a Twitter thread"""
        
        prompt = f"""
Create a compelling Twitter thread about this paper (8-12 tweets).
Each tweet should be under 280 characters.
Start with a hook, build tension, deliver insights, end with impact.

Title: {title}
Abstract: {abstract}
Key findings: {content[:1500]}

Format as JSON:
{{
  "thread": [
    {{"number": 1, "text": "...", "note": "[Description of visual if applicable]"}},
    ...
  ],
  "hashtags": ["#research", "..."],
  "engagement_tips": ["..."]
}}
"""
        
        response = self.llm.generate(
            prompt, temperature=0.7, max_tokens=1500, format_json=True
        )
        
        try:
            thread_data = json.loads(response)
        except:
            thread_data = {"thread": [], "hashtags": []}
        
        # Convert to readable format
        tweets_text = "\n\n".join([
            f"**Tweet {t['number']}:** {t['text']}"
            for t in thread_data.get("thread", [])
        ])
        
        if thread_data.get("hashtags"):
            tweets_text += f"\n\n**Hashtags:** {' '.join(thread_data['hashtags'])}"
        
        return ContentOutput(
            format="tweet_thread",
            title=f"Twitter Thread: {title}",
            content=tweets_text,
            metadata={
                "num_tweets": len(thread_data.get("thread", [])),
                "estimated_engagement": "High",
                "recommended_posting": "Business hours, Tuesday-Thursday"
            },
            export_formats={
                "markdown": tweets_text,
                "twitter_composer": json.dumps(thread_data),
                "scheduling_json": self._format_for_buffer(thread_data)
            }
        )
    
    def _generate_podcast_script(
        self,
        title: str,
        abstract: str,
        content: str,
        target_audience: Optional[str] = None
    ) -> ContentOutput:
        """Generate a podcast episode script"""
        
        audience = target_audience or "Researchers and practitioners"
        
        prompt = f"""
Write a podcast episode script about this paper.
Target audience: {audience}
Episode length: 20 minutes
Style: Conversational, but informative

Title: {title}
Abstract: {abstract}
Content: {content[:2000]}

Format as:

## INTRO (2 min)
[Host introduction, hook, episode outline]

## SEGMENT 1: THE PROBLEM (4 min)
[Why this research matters, context]

## SEGMENT 2: THE SOLUTION (7 min)
[How the authors approached it, key insights]

## SEGMENT 3: IMPACT (4 min)
[Implications, applications, future work]

## OUTRO (2 min)
[Wrap up, call to action, resources]

## DISCUSSION QUESTIONS
[3 questions for panel discussion]
"""
        
        script = self.llm.generate(prompt, temperature=0.7, max_tokens=2500)
        
        return ContentOutput(
            format="podcast_script",
            title=f"Podcast Script: {title}",
            content=script,
            metadata={
                "duration_minutes": 20,
                "target_audience": audience,
                "difficulty": "Intermediate",
                "production_notes": "Include paper link in show notes"
            },
            export_formats={
                "markdown": script,
                "html": self._markdown_to_html(script),
                "text": script
            }
        )
    
    def _generate_infographic_brief(
        self,
        title: str,
        abstract: str,
        content: str
    ) -> ContentOutput:
        """Generate brief for infographic designer using NER"""
        
        # 1. Extract technical entities using spaCy
        entities = extract_technical_entities(f"{title} {abstract} {content}")
        
        # 2. Format entities for the prompt
        entity_context = f"""
TECHNICAL ENTITIES TO FEATURE:
- Key Algorithms/Models: {', '.join(entities['algorithms']) or 'None extracted'}
- Datasets/Benchmarks: {', '.join(entities['datasets']) or 'None extracted'}
- Key Metrics: {', '.join(entities['metrics']) or 'None extracted'}
- Organizations/Authors: {', '.join(entities['organizations']) or 'None extracted'}
"""

        prompt = f"""
Create a brief for an infographic designer about this paper.
Include data points, key statistics, visual metaphors, and layout suggestions.

CRITICAL INSTRUCTION: You MUST create a dedicated section titled "TECHNOLOGIES & DATASETS USED" in your response and populate it directly using the TECHNICAL ENTITIES provided below.

Title: {title}
Abstract: {abstract}
Content: {content[:1500]}

{entity_context}

Format as:

## HEADLINE
[One powerful headline]

## KEY STATISTICS (3-5 data points)
- [stat and context]

## VISUAL METAPHORS
- [Metaphor 1]: Explains concept through visual similarity
- [Metaphor 2]: Another visual approach

## INFOGRAPHIC LAYOUT
[Suggested arrangement: top to bottom or left to right]

## COLOR PALETTE
[Suggested colors for different data categories]

## DATA SOURCES TO HIGHLIGHT
[Which facts should be prominently featured]
"""
        
        brief = self.llm.generate(prompt, temperature=0.6, max_tokens=1500)
        
        return ContentOutput(
            format="infographic_brief",
            title=f"Infographic Brief: {title}",
            content=brief,
            metadata={
                "target_format": "Infographic (1200x1800px or A4 landscape)",
                "tool_suggestions": ["Canva", "Figma", "Adobe Illustrator"],
                "designer_notes": "Brief is ready for visual design"
            },
            export_formats={
                "markdown": brief,
                "design_brief": brief
            }
        )
    
    def _generate_blog_post(
        self,
        title: str,
        abstract: str,
        content: str,
        target_audience: Optional[str] = None
    ) -> ContentOutput:
        """Generate a full blog post"""
        
        audience = target_audience or "Tech-savvy readers interested in AI/ML"
        
        prompt = f"""
Write a comprehensive blog post about this research paper.
Target audience: {audience}
Length: 1500-2000 words
Style: Engaging, informative, accessible to non-experts

Title: {title}
Abstract: {abstract}
Content: {content[:2500]}

Structure:
1. Hook/Introduction (200 words)
2. Background/Context (300 words)
3. Main Contribution (500 words)
4. Technical Deep Dive (500 words)
5. Real-World Applications (300 words)
6. Limitations and Future Work (200 words)
7. Conclusion (100 words)

Include:
- At least 2 subheadings per major section
- 1-2 code examples or pseudocode if applicable
- Links to related papers (use [Paper Name](link) format)
- Practical takeaways
"""
        
        blog_post = self.llm.generate(prompt, temperature=0.7, max_tokens=3000)
        
        return ContentOutput(
            format="blog_post",
            title=f"Blog Post: {title}",
            content=blog_post,
            metadata={
                "word_count": len(blog_post.split()),
                "reading_time_minutes": 8,
                "target_audience": audience,
                "seo_keywords": ["research", "paper", "ai", "machine learning"]
            },
            export_formats={
                "markdown": blog_post,
                "html": self._markdown_to_html(blog_post),
                "medium": blog_post,  # Can be pasted directly to Medium
                "dev_to": blog_post   # Can be pasted to Dev.to
            }
        )
    
    # Utility methods
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Simple markdown to HTML conversion (for demo)"""
        # In production, use a library like mistune or markdown
        html = markdown.replace("##", "<h2>").replace("**", "<strong>")
        return f"<div class='content'>{html}</div>"
    
    def _slides_to_markdown(self, slides: List[Dict]) -> str:
        """Convert slide data to markdown outline"""
        
        lines = []
        for slide in slides:
            lines.append(f"\n### Slide {slide.get('number', '?')}: {slide.get('title', '')}\n")
            
            for bullet in slide.get("bullets", []):
                lines.append(f"- {bullet}")
            
            if slide.get("notes"):
                lines.append(f"\n*Speaker notes: {slide['notes']}*")
        
        return "\n".join(lines)
    
    def _format_for_buffer(self, thread_data: Dict) -> str:
        """Format Twitter thread for Buffer scheduling"""
        
        return json.dumps({
            "posts": [
                {
                    "text": t["text"],
                    "shorten_links": True,
                    "add_image_description": t.get("note", "")
                }
                for t in thread_data.get("thread", [])
            ]
        }, indent=2)

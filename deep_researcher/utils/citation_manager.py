from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import re
from enum import Enum
import json
import hashlib

class CitationFormat(str, Enum):
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    BIBTEX = "bibtex"
    IEEE = "ieee"
    VANCOUVER = "vancouver"

class SourceType(str, Enum):
    ACADEMIC = "academic"
    LEGAL = "legal"
    MEDICAL = "medical"
    GOVERNMENT = "government"
    BUSINESS = "business"
    NEWS = "news"
    WEBSITE = "website"
    SOCIAL_MEDIA = "social_media"
    OTHER = "other"

class SourceCredibility(BaseModel):
    """Model for tracking source credibility metrics"""
    reliability_score: float = Field(description="Score from 0-1 indicating source reliability", ge=0, le=1)
    recency_score: float = Field(description="Score from 0-1 indicating how recent the source is", ge=0, le=1)
    authority_score: float = Field(description="Score from 0-1 indicating source authority", ge=0, le=1)
    overall_score: float = Field(description="Combined credibility score", ge=0, le=1)
    verification_status: str = Field(description="Status of source verification", default="unverified")
    verification_date: Optional[datetime] = Field(description="Date of last verification", default=None)
    verification_notes: Optional[str] = Field(description="Notes about verification process", default=None)

class Source(BaseModel):
    """Model for tracking source information"""
    url: str = Field(description="URL of the source")
    title: str = Field(description="Title of the source")
    author: Optional[str] = Field(description="Author of the source", default=None)
    publication_date: Optional[datetime] = Field(description="Publication date", default=None)
    publisher: Optional[str] = Field(description="Publisher of the source", default=None)
    credibility: SourceCredibility = Field(description="Credibility metrics for the source")
    citation_id: int = Field(description="Unique identifier for the citation")
    source_type: SourceType = Field(description="Type of source", default=SourceType.OTHER)
    industry: Optional[str] = Field(description="Industry or field this source belongs to", default=None)
    keywords: List[str] = Field(description="Keywords associated with this source", default_factory=list)
    metadata: Dict[str, Any] = Field(description="Additional metadata about the source", default_factory=dict)
    content_hash: Optional[str] = Field(description="Hash of the source content for verification", default=None)

class CitationManager:
    """Manages citations and sources for research reports"""
    
    def __init__(self):
        self.sources: Dict[int, Source] = {}
        self.next_citation_id = 1
        self.citation_format = CitationFormat.APA
        self.verified_domains: Dict[str, float] = {
            '.edu': 0.9,
            '.gov': 0.9,
            '.org': 0.8,
            '.com': 0.6,
            '.net': 0.6,
            '.io': 0.5
        }
        self.industry_weights: Dict[str, Dict[str, float]] = {
            'academic': {'reliability': 0.9, 'authority': 0.9},
            'legal': {'reliability': 0.8, 'authority': 0.8},
            'medical': {'reliability': 0.9, 'authority': 0.9},
            'government': {'reliability': 0.8, 'authority': 0.8},
            'business': {'reliability': 0.7, 'authority': 0.7},
            'news': {'reliability': 0.6, 'authority': 0.6}
        }

    def add_source(self, url: str, title: str, author: Optional[str] = None,
                  publication_date: Optional[datetime] = None, publisher: Optional[str] = None,
                  source_type: SourceType = SourceType.OTHER, industry: Optional[str] = None,
                  keywords: List[str] = None, metadata: Dict[str, Any] = None) -> int:
        """Add a new source and return its citation ID"""
        source = Source(
            url=url,
            title=title,
            author=author,
            publication_date=publication_date,
            publisher=publisher,
            citation_id=self.next_citation_id,
            credibility=self._calculate_credibility(url, publication_date, source_type, industry),
            source_type=source_type,
            industry=industry,
            keywords=keywords or [],
            metadata=metadata or {},
            content_hash=self._generate_content_hash(url, title, author)
        )
        self.sources[self.next_citation_id] = source
        citation_id = self.next_citation_id
        self.next_citation_id += 1
        return citation_id

    def _generate_content_hash(self, url: str, title: str, author: Optional[str]) -> str:
        """Generate a hash of the source content for verification"""
        content = f"{url}{title}{author or ''}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _calculate_credibility(self, url: str, publication_date: Optional[datetime],
                             source_type: SourceType, industry: Optional[str]) -> SourceCredibility:
        """Calculate credibility scores for a source"""
        # Base scores
        reliability_score = 0.7
        recency_score = 0.7
        authority_score = 0.7

        # Adjust recency score based on publication date
        if publication_date:
            years_old = (datetime.now() - publication_date).days / 365
            recency_score = max(0, 1 - (years_old / 10))

        # Adjust reliability and authority scores based on domain
        domain = url.lower()
        for d, score in self.verified_domains.items():
            if d in domain:
                reliability_score = max(reliability_score, score)
                authority_score = max(authority_score, score)
                break

        # Adjust scores based on source type
        if source_type == SourceType.ACADEMIC:
            reliability_score = max(reliability_score, 0.9)
            authority_score = max(authority_score, 0.9)
        elif source_type == SourceType.GOVERNMENT:
            reliability_score = max(reliability_score, 0.8)
            authority_score = max(authority_score, 0.8)
        elif source_type == SourceType.MEDICAL:
            reliability_score = max(reliability_score, 0.9)
            authority_score = max(authority_score, 0.9)

        # Adjust scores based on industry
        if industry and industry in self.industry_weights:
            weights = self.industry_weights[industry]
            reliability_score = max(reliability_score, weights['reliability'])
            authority_score = max(authority_score, weights['authority'])

        overall_score = (reliability_score + recency_score + authority_score) / 3

        return SourceCredibility(
            reliability_score=reliability_score,
            recency_score=recency_score,
            authority_score=authority_score,
            overall_score=overall_score
        )

    def verify_source(self, citation_id: int, verification_notes: str) -> None:
        """Verify a source and update its credibility"""
        if citation_id in self.sources:
            source = self.sources[citation_id]
            source.credibility.verification_status = "verified"
            source.credibility.verification_date = datetime.now()
            source.credibility.verification_notes = verification_notes
            # Increase credibility scores for verified sources
            source.credibility.reliability_score = min(1.0, source.credibility.reliability_score + 0.1)
            source.credibility.authority_score = min(1.0, source.credibility.authority_score + 0.1)
            source.credibility.overall_score = (source.credibility.reliability_score + 
                                             source.credibility.recency_score + 
                                             source.credibility.authority_score) / 3

    def export_sources(self, format: str = "json") -> str:
        """Export sources in various formats"""
        if format == "json":
            return json.dumps([source.model_dump() for source in self.sources.values()], 
                            default=str, indent=2)
        elif format == "bibtex":
            return self._generate_bibtex()
        elif format == "csv":
            return self._generate_csv()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _generate_bibtex(self) -> str:
        """Generate BibTeX format for sources"""
        bibtex_entries = []
        for source in self.sources.values():
            entry = f"@misc{{{source.citation_id},\n"
            entry += f"  title = {{{source.title}}},\n"
            if source.author:
                entry += f"  author = {{{source.author}}},\n"
            if source.publication_date:
                entry += f"  year = {{{source.publication_date.year}}},\n"
            if source.publisher:
                entry += f"  publisher = {{{source.publisher}}},\n"
            entry += f"  url = {{{source.url}}},\n"
            entry += "}\n"
            bibtex_entries.append(entry)
        return "\n".join(bibtex_entries)

    def _generate_csv(self) -> str:
        """Generate CSV format for sources"""
        headers = ["citation_id", "title", "author", "publication_date", "publisher", 
                  "url", "source_type", "industry", "reliability_score", "authority_score"]
        rows = [headers]
        for source in self.sources.values():
            row = [
                str(source.citation_id),
                source.title,
                source.author or "",
                source.publication_date.strftime("%Y-%m-%d") if source.publication_date else "",
                source.publisher or "",
                source.url,
                source.source_type,
                source.industry or "",
                str(source.credibility.reliability_score),
                str(source.credibility.authority_score)
            ]
            rows.append(row)
        return "\n".join([",".join(row) for row in rows])

    def get_sources_by_industry(self, industry: str) -> List[Source]:
        """Get all sources for a specific industry"""
        return [source for source in self.sources.values() if source.industry == industry]

    def get_sources_by_type(self, source_type: SourceType) -> List[Source]:
        """Get all sources of a specific type"""
        return [source for source in self.sources.values() if source.source_type == source_type]

    def get_high_credibility_sources(self, threshold: float = 0.8) -> List[Source]:
        """Get all sources with credibility above threshold"""
        return [source for source in self.sources.values() 
                if source.credibility.overall_score >= threshold]

    def get_recent_sources(self, days: int = 365) -> List[Source]:
        """Get all sources published within the last n days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [source for source in self.sources.values() 
                if source.publication_date and source.publication_date >= cutoff_date]

    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about the sources"""
        stats = {
            "total_sources": len(self.sources),
            "sources_by_type": {},
            "sources_by_industry": {},
            "average_credibility": 0.0,
            "verified_sources": 0,
            "recent_sources": 0
        }
        
        if not self.sources:
            return stats

        # Calculate statistics
        total_credibility = 0
        cutoff_date = datetime.now() - timedelta(days=365)
        
        for source in self.sources.values():
            # Count by type
            stats["sources_by_type"][source.source_type] = \
                stats["sources_by_type"].get(source.source_type, 0) + 1
            
            # Count by industry
            if source.industry:
                stats["sources_by_industry"][source.industry] = \
                    stats["sources_by_industry"].get(source.industry, 0) + 1
            
            # Track credibility
            total_credibility += source.credibility.overall_score
            
            # Count verified sources
            if source.credibility.verification_status == "verified":
                stats["verified_sources"] += 1
            
            # Count recent sources
            if source.publication_date and source.publication_date >= cutoff_date:
                stats["recent_sources"] += 1
        
        stats["average_credibility"] = total_credibility / len(self.sources)
        
        return stats

    def format_citation(self, citation_id: int, format: Optional[CitationFormat] = None) -> str:
        """Format a citation according to the specified style"""
        if citation_id not in self.sources:
            return f"[{citation_id}]"

        source = self.sources[citation_id]
        format = format or self.citation_format

        if format == CitationFormat.APA:
            return self._format_apa(source)
        elif format == CitationFormat.MLA:
            return self._format_mla(source)
        elif format == CitationFormat.CHICAGO:
            return self._format_chicago(source)
        elif format == CitationFormat.HARVARD:
            return self._format_harvard(source)
        elif format == CitationFormat.BIBTEX:
            return self._format_bibtex(source)
        elif format == CitationFormat.IEEE:
            return self._format_ieee(source)
        elif format == CitationFormat.VANCOUVER:
            return self._format_vancouver(source)
        else:
            return f"[{citation_id}]"

    def _format_apa(self, source: Source) -> str:
        """Format citation in APA style"""
        parts = []
        if source.author:
            parts.append(source.author)
        if source.publication_date:
            parts.append(f"({source.publication_date.year})")
        parts.append(source.title)
        if source.publisher:
            parts.append(source.publisher)
        return ", ".join(parts)

    def _format_mla(self, source: Source) -> str:
        """Format citation in MLA style"""
        parts = []
        if source.author:
            parts.append(source.author)
        parts.append(f'"{source.title}"')
        if source.publisher:
            parts.append(source.publisher)
        if source.publication_date:
            parts.append(str(source.publication_date.year))
        return ", ".join(parts)

    def _format_chicago(self, source: Source) -> str:
        """Format citation in Chicago style"""
        parts = []
        if source.author:
            parts.append(source.author)
        parts.append(source.title)
        if source.publisher:
            parts.append(source.publisher)
        if source.publication_date:
            parts.append(str(source.publication_date.year))
        return ", ".join(parts)

    def _format_harvard(self, source: Source) -> str:
        """Format citation in Harvard style"""
        parts = []
        if source.author:
            parts.append(source.author)
        if source.publication_date:
            parts.append(f"({source.publication_date.year})")
        parts.append(source.title)
        if source.publisher:
            parts.append(source.publisher)
        return ", ".join(parts)

    def _format_bibtex(self, source: Source) -> str:
        """Format citation in BibTeX style"""
        parts = []
        if source.author:
            parts.append(f"author = {{{source.author}}}")
        if source.publication_date:
            parts.append(f"year = {{{source.publication_date.year}}}")
        parts.append(f"title = {{{source.title}}}")
        if source.publisher:
            parts.append(f"publisher = {{{source.publisher}}}")
        parts.append(f"url = {{{source.url}}}")
        return "@misc{" + ", ".join(parts) + "}"

    def _format_ieee(self, source: Source) -> str:
        """Format citation in IEEE style"""
        parts = []
        if source.author:
            parts.append(source.author)
        if source.publication_date:
            parts.append(f"({source.publication_date.year})")
        parts.append(source.title)
        if source.publisher:
            parts.append(source.publisher)
        return ", ".join(parts)

    def _format_vancouver(self, source: Source) -> str:
        """Format citation in Vancouver style"""
        parts = []
        if source.author:
            parts.append(source.author)
        if source.publication_date:
            parts.append(f"({source.publication_date.year})")
        parts.append(source.title)
        if source.publisher:
            parts.append(source.publisher)
        return ", ".join(parts)

    def generate_bibliography(self, format: Optional[CitationFormat] = None) -> str:
        """Generate a bibliography of all sources"""
        format = format or self.citation_format
        bibliography = []
        
        for citation_id in sorted(self.sources.keys()):
            source = self.sources[citation_id]
            citation = self.format_citation(citation_id, format)
            bibliography.append(f"[{citation_id}] {citation}")
        
        return "\n".join(bibliography)

    def get_source_credibility_summary(self) -> str:
        """Generate a summary of source credibility"""
        if not self.sources:
            return "No sources available."

        total_sources = len(self.sources)
        avg_reliability = sum(s.credibility.reliability_score for s in self.sources.values()) / total_sources
        avg_recency = sum(s.credibility.recency_score for s in self.sources.values()) / total_sources
        avg_authority = sum(s.credibility.authority_score for s in self.sources.values()) / total_sources
        avg_overall = sum(s.credibility.overall_score for s in self.sources.values()) / total_sources

        return f"""Source Credibility Summary:
Total Sources: {total_sources}
Average Reliability Score: {avg_reliability:.2f}
Average Recency Score: {avg_recency:.2f}
Average Authority Score: {avg_authority:.2f}
Overall Average Score: {avg_overall:.2f}""" 
"""
validation.py

Article validation and citation accuracy subsystems for MicroTraitLLM.

This module implements:
- Article validation (recency, journal reputation, citation count, topic relevance)
- Citation accuracy checking (mapping citations to identifiers and verifying against retrieved corpus)

Public entry points:
- validate_articles(query, articles)
- check_citations(answer_body, ref_text, retrieved_articles, similarity_threshold=0.6)

TODO:
- Wire `embed_text` into the actual embedding model used by the retriever.
- Replace `check_identifier_accessibility` stub with real NCBI/PMC or internal metadata calls.
- Integrate `validate_articles` and `check_citations` into the main MTLLM RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import math
import re
import logging

logger = logging.getLogger(__name__)

CURRENT_YEAR = 2025

# Simple journal tier map. Extend this based on real data or an external config.
JOURNAL_TIERS: Dict[str, float] = {
    "Nature Microbiology": 1.0,
    "Science": 1.0,
    "Cell": 1.0,
    "PNAS": 0.9,
    # TODO: add more journals / import from config or metadata store
}

DEFAULT_PEER_REVIEWED = 0.7
PREPRINT_REPO_SCORE = 0.3

# Regex patterns for citations and identifiers
CITATION_PATTERN = re.compile(r"\[(\d+)\]")          # e.g. [1], [2]
REF_ENTRY_PATTERN = re.compile(r"^\[(\d+)\]\s+(.*)$")  # reference list line
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
PMCID_PATTERN = re.compile(r"\bPMC\d+\b", re.IGNORECASE)


@dataclass
class Article:
    """
    Representation of a retrieved article in the MTLLM pipeline.

    Attributes:
        pmcid: PMCID string, e.g. "PMC1234567", if available.
        doi: DOI string, if available.
        title: Article title.
        abstract: Article abstract or summary text.
        journal: Journal or source name.
        year: Year of publication.
        citation_count: Approximate citation count (can be 0 if unavailable).
        is_peer_reviewed: Whether the article is known to be peer-reviewed.
        is_retracted: True if the article has been retracted.
        validation_score: Composite score assigned during validation.
        confidence_label: Categorical label ("high", "medium", "low", "invalid_id", "unknown").
    """
    pmcid: Optional[str]
    doi: Optional[str]
    title: str
    abstract: str
    journal: str
    year: int
    citation_count: int
    is_peer_reviewed: bool
    is_retracted: bool

    # filled by validator
    validation_score: float = 0.0
    confidence_label: str = "unknown"


# ===== Embedding + similarity utilities =====================================

def embed_text(text: str):
    """
    Return an embedding vector for the given text.

    TODO:
        - Replace this stub with the actual embedding model call used by MTLLM.
        - Ensure this is consistent with the retriever's embedding space.
    """
    raise NotImplementedError("embed_text must be wired to the actual embedding model.")


def cosine_similarity(a, b) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector (list or iterable of floats).
        b: Second embedding vector.

    Returns:
        Cosine similarity in [0, 1] (values < 0 are clamped to 0).
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, sim))


# ===== Article validation scoring functions ==================================

def recency_score(year: int) -> float:
    """
    Score recency on [0, 1] where 1 = current year, 0 = >=10 years old.
    """
    age = CURRENT_YEAR - year
    score = max(0.0, 1.0 - age / 10.0)
    logger.debug("Recency score for year %s (age=%s): %s", year, age, score)
    return score


def reputation_score(article: Article) -> float:
    """
    Score journal/source reputation.

    Rules:
        - Retracted -> 0.0
        - Known journal in JOURNAL_TIERS -> tier score
        - Peer-reviewed but unknown journal -> DEFAULT_PEER_REVIEWED
        - Otherwise (e.g., preprint/unknown) -> PREPRINT_REPO_SCORE
    """
    if article.is_retracted:
        logger.debug("Article %s flagged as retracted.", article.pmcid or article.doi or article.title)
        return 0.0
    if article.journal in JOURNAL_TIERS:
        return JOURNAL_TIERS[article.journal]
    if article.is_peer_reviewed:
        return DEFAULT_PEER_REVIEWED
    return PREPRINT_REPO_SCORE


def citation_score(c: int, c_min: int, c_max: int) -> float:
    """
    Score citation count using log-scaled min-max normalization.

    Args:
        c: citation count for this article
        c_min: minimum citation count in the candidate set
        c_max: maximum citation count in the candidate set
    """
    if c_max <= c_min:
        return 0.0
    num = math.log1p(c) - math.log1p(c_min)
    den = math.log1p(c_max) - math.log1p(c_min)
    score = max(0.0, min(1.0, num / den))
    logger.debug("Citation score for c=%s (min=%s, max=%s): %s", c, c_min, c_max, score)
    return score


def topic_relevance_score(query: str, article: Article) -> float:
    """
    Compute topic relevance as cosine similarity between query and article text.
    """
    q_emb = embed_text(query)
    d_emb = embed_text(article.title + " " + article.abstract)
    score = cosine_similarity(q_emb, d_emb)
    logger.debug("Topic relevance for article %s: %s", article.pmcid or article.doi or article.title, score)
    return score


def check_identifier_accessibility(pmcid: Optional[str], doi: Optional[str]) -> bool:
    """
    Check whether a PMCID/DOI is valid and accessible.

    TODO:
        - Implement real checks using NCBI/PMC APIs or a local metadata cache.
        - Handle request failures / timeouts gracefully.

    For now, this stub returns True so that the rest of the pipeline can be exercised.
    """
    # Example future implementation idea:
    # - Query NCBI Entrez for pmcid or doi
    # - Verify status is not "retracted" and is reachable
    return True


def validate_articles(query: str, articles: List[Article]) -> List[Article]:
    """
    Validate and score retrieved articles for use in MicroTraitLLM.

    The composite score is a weighted combination of:
        - Recency
        - Source reputation
        - Citation count
        - Topic relevance

    Args:
        query: User query string.
        articles: List of Article objects retrieved by the RAG retriever.

    Returns:
        A filtered list of Article objects with updated `validation_score`
        and `confidence_label`. By default, "low" confidence articles are dropped.
    """
    if not articles:
        logger.info("validate_articles called with empty article list.")
        return []

    c_counts = [a.citation_count for a in articles]
    c_min, c_max = min(c_counts), max(c_counts)

    validated: List[Article] = []

    for a in articles:
        if not check_identifier_accessibility(a.pmcid, a.doi):
            logger.warning("Identifier check failed for article %s.", a.pmcid or a.doi or a.title)
            a.validation_score = 0.0
            a.confidence_label = "invalid_id"
            continue

        s_r = recency_score(a.year)
        s_rep = reputation_score(a)
        s_c = citation_score(a.citation_count, c_min, c_max)
        s_t = topic_relevance_score(query, a)

        # TODO: allow weights to be configurable.
        score = 0.25 * s_r + 0.25 * s_rep + 0.20 * s_c + 0.30 * s_t
        a.validation_score = score

        if score >= 0.75:
            a.confidence_label = "high"
        elif score >= 0.5:
            a.confidence_label = "medium"
        else:
            a.confidence_label = "low"

        logger.debug(
            "Article %s scored %.3f (recency=%.3f, rep=%.3f, citations=%.3f, topic=%.3f) -> %s",
            a.pmcid or a.doi or a.title,
            score,
            s_r,
            s_rep,
            s_c,
            s_t,
            a.confidence_label,
        )

        validated.append(a)

    # By default, drop low-confidence articles
    filtered = [a for a in validated if a.confidence_label in ("high", "medium")]
    logger.info("validate_articles returning %d/%d articles.", len(filtered), len(validated))
    return filtered


# ===== Citation accuracy utilities ==========================================

def extract_numeric_citations(answer: str) -> List[str]:
    """
    Extract unique numeric citation labels (e.g. '1', '2') from an answer body.
    """
    return list(sorted(set(CITATION_PATTERN.findall(answer))))


def parse_reference_list(ref_text: str) -> Dict[str, str]:
    """
    Parse a reference list into a mapping: ref_number -> raw reference string.

    Expects lines like:
        [1] Author et al. Title...
    """
    mapping: Dict[str, str] = {}
    for line in ref_text.splitlines():
        m = REF_ENTRY_PATTERN.match(line.strip())
        if not m:
            continue
        num, content = m.groups()
        mapping[num] = content
    return mapping


def extract_identifier_from_reference(ref_entry: str) -> Optional[str]:
    """
    Extract a PMCID or DOI from a raw reference entry string, if present.
    """
    pmcid_match = PMCID_PATTERN.search(ref_entry)
    if pmcid_match:
        return pmcid_match.group(0)
    doi_match = DOI_PATTERN.search(ref_entry)
    if doi_match:
        return doi_match.group(0)
    return None


def build_citation_identifier_map(ref_text: str) -> Dict[str, str]:
    """
    Build a mapping from reference number (as string) to an identifier (PMCID or DOI).
    """
    ref_entries = parse_reference_list(ref_text)
    id_map: Dict[str, str] = {}
    for num, entry in ref_entries.items():
        identifier = extract_identifier_from_reference(entry)
        if identifier:
            id_map[num] = identifier
        else:
            logger.debug("No identifier found for reference [%s]: %s", num, entry)
    return id_map


def find_article_by_identifier(identifier: str, articles: List[Article]) -> Optional[Article]:
    """
    Find an article in `articles` matching the given PMCID or DOI identifier.
    """
    identifier_norm = identifier.strip()
    for a in articles:
        if a.pmcid and identifier_norm.upper() == a.pmcid.upper():
            return a
        if a.doi and identifier_norm.lower() == a.doi.lower():
            return a
    return None


def get_context_snippet(answer: str, raw_token: str, window: int = 200) -> str:
    """
    Grab ~`window` characters around the citation token.

    Args:
        answer: Full answer text containing citations.
        raw_token: Citation token string, e.g. "[1]".
        window: Number of characters to include on each side.

    Returns:
        A context substring, or empty string if the token is not found.
    """
    idx = answer.find(raw_token)
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(answer), idx + len(raw_token) + window)
    return answer[start:end]


def similarity_to_article(snippet: str, article: Article) -> float:
    """
    Compute similarity between an answer snippet and an article's text.
    """
    s_emb = embed_text(snippet)
    a_emb = embed_text(article.title + " " + article.abstract)
    return cosine_similarity(s_emb, a_emb)


def check_citations(
    answer_body: str,
    ref_text: str,
    retrieved_articles: List[Article],
    similarity_threshold: float = 0.6,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Check and clean citations in a model-generated answer.

    Steps:
        1. Extract numeric citations in the answer body (e.g. [1], [2]).
        2. Map each number to a PMCID/DOI using the reference list.
        3. Match identifiers to retrieved articles.
        4. Compute embedding-based similarity between local context and the article.
        5. Remove or flag hallucinated / mismatched citations.

    Args:
        answer_body: The main answer text (excluding the reference list).
        ref_text: The raw reference list text produced by the model.
        retrieved_articles: Articles retrieved in the RAG pipeline.
        similarity_threshold: Minimum similarity for a citation to be considered valid.

    Returns:
        cleaned_answer_body: Answer text with hallucinated citations removed.
        report_list: List of diagnostic dicts per citation, each containing:
            - raw_citation: e.g. "[1]"
            - identifier: PMCID/DOI or None
            - status: "valid", "mismatch", or "not_found"
            - similarity: similarity score (0.0 if not computed)
    """
    cited_nums = extract_numeric_citations(answer_body)
    num_to_identifier = build_citation_identifier_map(ref_text)

    report: List[Dict[str, Any]] = []
    cleaned_answer = answer_body

    for num in cited_nums:
        raw_token = f"[{num}]"
        identifier = num_to_identifier.get(num)
        status = "not_found"
        sim = 0.0

        article = None
        if identifier:
            article = find_article_by_identifier(identifier, retrieved_articles)

        if article is None:
            # No matching article in corpus => hallucinated reference
            logger.warning("No article found for citation %s (identifier=%s).", raw_token, identifier)
            cleaned_answer = cleaned_answer.replace(raw_token, "")
        else:
            snippet = get_context_snippet(answer_body, raw_token)
            if snippet:
                sim = similarity_to_article(snippet, article)
                if sim >= similarity_threshold:
                    status = "valid"
                else:
                    status = "mismatch"
                    logger.info(
                        "Citation %s mismatch: similarity=%.3f below threshold=%.3f.",
                        raw_token,
                        sim,
                        similarity_threshold,
                    )
            else:
                logger.info("No context snippet found for citation %s.", raw_token)

        report.append(
            {
                "raw_citation": raw_token,
                "identifier": identifier,
                "status": status,
                "similarity": sim,
            }
        )

    return cleaned_answer, report
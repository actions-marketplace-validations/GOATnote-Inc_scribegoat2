"""
ScribeGoat2 Co-Author Persona Tools

This package provides tools for maintaining the Enhanced Co-Author Persona:
- commit_generator: Standardized commit message generation
- next_best_action: Prioritized action suggestions
"""

from .commit_generator import CommitMessage, CommitType, Scope
from .next_best_action import Action, Category, Priority, RepoAnalyzer

__all__ = [
    "CommitMessage",
    "CommitType",
    "Scope",
    "RepoAnalyzer",
    "Action",
    "Priority",
    "Category",
]

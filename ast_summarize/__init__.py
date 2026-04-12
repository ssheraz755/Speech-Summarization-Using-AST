"""AST-based speed summarization via attention-driven frame selection."""

from .pipeline import ASTSummarizeResult, ast_speed_summarize, load_ast_model

__all__ = ["ASTSummarizeResult", "ast_speed_summarize", "load_ast_model"]

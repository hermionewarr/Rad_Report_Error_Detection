"""Backward-compatible re-exports for predictions_eval.

New code should import from src.predictions_eval directly.
"""
try:
    from src.predictions_eval import *  # noqa: F401,F403
    from src.predictions_eval import main
except ImportError:
    from predictions_eval import *  # type: ignore  # noqa: F401,F403
    from predictions_eval import main  # type: ignore

if __name__ == "__main__":
    main()

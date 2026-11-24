import sys
from pathlib import Path

sys.path.insert(0, "/tests/domain")
from victors_werkbank.grading.grader import grade_uploaded

grade_uploaded(Path("/trusted"), Path("/logs/verifier"))

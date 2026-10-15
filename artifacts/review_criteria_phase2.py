"""
Phase 2 Module Review Criteria
==============================

Structured checklist for evaluating the implementation of Phase 2 core modules:
config.py, metadata.py, errors.py, scanner.py

Based on:
- artifacts/tasks.md (T2.1 - T2.4)
- artifacts/design.md (sections 3.2, 3.3)

Usage:
    Review each module against the criteria below. Mark items as:
      [ ] = Not reviewed / Not implemented
      [~] = Partially implemented / Needs work
      [x] = Fully implemented and verified

Review Status Legend is embedded in each section.
"""

# =============================================================================
# T2.1: Configuration System (config.py)
# =============================================================================

T2_1_CHECKLIST = """
T2.1: Configuration System (config.py)

1. FUNCTIONALITY CHECKS
   Environment Variables (Design Doc §3.2):
   [ ] BASE_URL - required, type str, no default
   [ ] DATA_DIR - required, type str, no default
   [ ] LLM_ENDPOINT - required, type str, no default
   [ ] LLM_API_KEY - required, type str, no default
   [ ] LLM_MODEL - optional, type str, default 'mistralai/Mistral-7B-Instruct'
   [ ] FEED_FORMAT - optional, type str, default 'rss', validated as 'rss' or 'atom'
   [ ] TRANSCRIPTION_MODEL - optional, type str, default 'large-v3'
   [ ] TRANSCRIPTION_DEVICE - optional, type str, default 'cuda'
   [ ] TRANSCRIPTION_COMPUTE_TYPE - optional, type str, default 'float16'
   [ ] LLM_SYSTEM_PROMPT - optional, type str, default from design doc §3.2

   Settings Class:
   [ ] Settings class extends pydantic_settings.BaseSettings
   [ ] All env vars are defined as class attributes with correct types
   [ ] Required fields raise ValidationError when missing
   [ ] Optional fields fall back to defaults when unset
   [ ] model_config or Config class sets env_file='.env' and env_file_encoding='utf-8'

   Singleton Pattern:
   [ ] A module-level singleton accessor function exists (e.g., get_settings())
   [ ] Singleton caches the Settings instance on first call
   [ ] Subsequent calls return the cached instance
   [ ] No global mutable state is exposed beyond the singleton

   Loading & Validation:
   [ ] Config loads successfully from .env file when present
   [ ] Config loads successfully from environment variables when .env is absent
   [ ] Missing required fields produce clear ValidationError messages
   [ ] DATA_DIR path is validated to exist (or warning is logged if not)
   [ ] BASE_URL is validated as a well-formed URL (optional but recommended)

2. DESIGN ALIGNMENT
   [ ] Configuration is the single source of truth for all runtime parameters
   [ ] No hardcoded values for env vars that should be configurable
   [ ] Defaults match design doc §3.2 exactly
   [ ] FEED_FORMAT constraint (rss only per design doc §10.1) is enforced or documented
   [ ] System prompt default matches design doc §10.3 exactly

3. CODE QUALITY
   [ ] Class and all fields have docstrings
   [ ] Type annotations are used throughout (Python 3.14+ syntax)
   [ ] No unnecessary complexity or boilerplate
   [ ] Imports are minimal and correct
   [ ] Naming follows Python conventions (PascalCase for classes, UPPER_CASE for env var names)

4. ERROR HANDLING
   [ ] Missing required fields raise pydantic.ValidationError with clear messages
   [ ] Invalid values (e.g., FEED_FORMAT=invalid) are rejected at load time
   [ ] Settings initialization surfaces errors to the caller
   [ ] No silent failures or empty defaults for required fields

5. TESTING READINESS
   [ ] Settings can be instantiated with explicit kwargs for unit tests
   [ ] Settings can be instantiated from a dict (model_validate)
   [ ] Singleton can be reset or overridden for test isolation
   [ ] .env file loading can be disabled or overridden for tests
   [ ] Validation error messages are deterministic and testable
"""

# =============================================================================
# T2.2: Metadata Model (metadata.py)
# =============================================================================

T2_2_CHECKLIST = """
T2.2: Metadata Model (metadata.py)

1. FUNCTIONALITY CHECKS
    Schema Fields (Design Doc §3.3):
    [ ] transcribe_status - str | None, 'in_progress' or 'complete' (MP4 only)
    [ ] summarize_status - str | None, 'in_progress' or 'complete' (MP4 only)
    [ ] extract_status - str | None, 'in_progress' or 'complete' (PDF/HTML only)
    [ ] title - str, required
    [ ] description - str, required
    [ ] filename - str, required (original filename with extension)
    [ ] filepath - str, required (relative path from DATA_DIR root)
    [ ] filetype - str, must be 'mp4', 'pdf', or 'html'
    [ ] date - str (ISO8601), required
    [ ] size - int (bytes), required
    [ ] transcript - str | None, only for filetype='mp4', optional
    [ ] summary - str | None, only for filetype='mp4', optional
    [ ] processing_date - str (ISO8601), required (auto-set if not provided)
    [ ] stage - str | None, optional (current processing stage)
    [ ] transcription_model - str | None, optional (MP4 only)
    [ ] started_at - str (ISO8601) | None, optional
    [ ] completed_at - str (ISO8601) | None, optional
    [ ] error - str | None, only present when processing failed

    Validation Rules:
    [ ] filetype field is validated to be 'mp4', 'pdf', or 'html' only
    [ ] error field is only present when processing failed
    [ ] transcript and summary fields are only valid for filetype='mp4'
    [ ] transcribe_status and summarize_status are only valid for filetype='mp4'
    [ ] extract_status is only valid for filetype='pdf' or 'html'
    [ ] date and processing_date are validated as ISO8601 format
    [ ] size is validated as a non-negative integer
    [ ] filepath does not contain path traversal sequences (../, absolute paths)
    [ ] feed_eligible is a computed property based on step statuses

   Serialization / Deserialization:
   [ ] to_dict() method returns a dict matching the exact JSON schema
   [ ] from_dict() classmethod creates a Metadata instance from a dict
   [ ] to_json() method serializes to a JSON string
   [ ] from_json() classmethod parses a JSON string into a Metadata instance
   [ ] Optional fields (transcript, summary, error) are omitted from JSON when None
   [ ] JSON output matches the schema in design doc §3.3 exactly

   File I/O Methods:
   [ ] write_to_file(path: str | Path) method writes JSON to the specified path
   [ ] read_from_file(path: str | Path) classmethod reads and validates JSON
   [ ] File I/O methods handle FileNotFoundError and json.JSONDecodeError gracefully
   [ ] File paths are validated to prevent directory traversal

2. DESIGN ALIGNMENT
   [ ] Metadata is a Pydantic model (per design doc §2.2)
    [ ] Schema matches design doc §3.3 exactly (all fields, correct names and order)
   [ ] filepath is relative to DATA_DIR root (not absolute)
   [ ] size is in bytes (for podcast enclosure length)
   [ ] error entries are excluded from feeds (design doc §7.1)
   [ ] transcript and summary are MP4-only fields (design doc §3.3)

3. CODE QUALITY
   [ ] Pydantic model with Field() definitions for all fields
   [ ] All fields have descriptions or docstrings
   [ ] Type annotations use modern Python 3.14+ syntax (str | None not Optional[str])
   [ ] Model docstring explains the schema and its purpose
   [ ] No redundant code; leverages Pydantic's built-in validation

4. ERROR HANDLING
    [ ] Invalid filetype value raises ValidationError with clear message
    [ ] error field present with non-complete statuses raises ValidationError
    [ ] transcript/summary present with non-MP4 filetype raises ValidationError
    [ ] Invalid ISO8601 date raises ValidationError
    [ ] Negative size raises ValidationError
    [ ] File I/O errors are caught and wrapped in appropriate exceptions

5. TESTING READINESS
    [ ] Model can be instantiated with all valid field combinations
    [ ] Model rejects all invalid field combinations with clear errors
    [ ] to_json() / from_json() round-trip preserves all data
    [ ] to_dict() / from_dict() round-trip preserves all data
    [ ] Error metadata can be created and validated (error='message')
    [ ] Tests can construct Metadata instances with minimal required fields
    [ ] Default processing_date is auto-populated
"""

# =============================================================================
# T2.4: Error Handling Utilities (errors.py)
# =============================================================================

T2_4_CHECKLIST = """
T2.4: Error Handling Utilities (errors.py)

1. FUNCTIONALITY CHECKS
   Custom Exception Classes:
   [ ] ProcessingError - base exception for all processing errors
   [ ] TranscriptionError - inherits from ProcessingError, for transcription failures
   [ ] SummarizationError - inherits from ProcessingError, for LLM/summarization failures
   [ ] ValidationError - inherits from ProcessingError, for metadata/schema validation errors
   [ ] All exceptions accept a message parameter and optional details dict
   [ ] All exceptions have a __str__ method that returns a clear error message

    Error Metadata Creation:
    [ ] create_error_metadata(exception, filename, filepath) -> Metadata
    [ ] Returns a Metadata instance with error set to exception message
    [ ] Sets all step statuses to None
    [ ] Sets filetype based on the file extension in filename
    [ ] Sets processing_date to the current timestamp
    [ ] Handles all custom exception types (ProcessingError, TranscriptionError, etc.)
    [ ] Handles generic Exception as a fallback

   Error Logging Helper:
   [ ] log_error(exception, context='') function or similar
   [ ] Log format includes: timestamp, file/module name, error message
   [ ] Uses Python logging module (not print())
   [ ] Log level is ERROR for all exception types
   [ ] Log message includes context information (e.g., which file being processed)
   [ ] Log format is consistent across all error logging calls

2. DESIGN ALIGNMENT
   [ ] Exceptions are organized in a hierarchy (base ProcessingError with subclasses)
   [ ] Error metadata creation is reusable across all processing modules
   [ ] Logging format is consistent with the rest of the codebase
   [ ] Error handling strategy matches design doc §7.1
   [ ] No errors are silently swallowed

3. CODE QUALITY
   [ ] All exception classes have docstrings explaining when they should be raised
   [ ] create_error_metadata has a clear docstring with parameters and return type
   [ ] log_error has a clear docstring
   [ ] Type annotations are used throughout
   [ ] No circular imports (errors.py should not import from producer.py, etc.)

4. ERROR HANDLING
   [ ] create_error_metadata handles exceptions with no message gracefully
   [ ] create_error_metadata truncates very long error messages (optional)
   [ ] log_error handles exceptions that cannot be converted to string
   [ ] Logging does not raise exceptions itself

5. TESTING READINESS
   [ ] Each exception class can be raised and caught by type
   [ ] create_error_metadata produces valid Metadata instances for all exception types
   [ ] create_error_metadata output can be serialized to JSON
   [ ] log_error can be tested with caplog or mock logging
   [ ] Exception messages are deterministic and testable
   [ ] Error metadata error field contains the original exception message
"""

# =============================================================================
# T2.3: File Scanner (scanner.py)
# =============================================================================

T2_3_CHECKLIST = """
T2.3: File Scanner (scanner.py)

1. FUNCTIONALITY CHECKS
   Scanner Class:
   [ ] Scanner class with __init__(data_dir: str | Path) constructor
   [ ] data_dir is stored as an absolute Path object

   scan_directory() Method:
   [ ] Recursively walks DATA_DIR using os.walk or pathlib.Path.rglob
   [ ] Finds files with extensions .mp4, .pdf, .html (case-insensitive)
   [ ] For each found file, checks if corresponding .json file exists in same directory
   [ ] JSON filename is constructed as filename + '.json' (e.g., lecture.mp4.json)
   [ ] Returns list of files that need processing:
     [ ] Files with no .json file present
     [ ] Files where .json exists but is 'outdated' (source modified after JSON)
   [ ] Returns absolute file paths for source files
   [ ] Returns relative paths from DATA_DIR root
   [ ] Return type is a list of dataclass/namedtuple/dict with absolute_path and relative_path fields

   Filename Validation (Special Characters):
   [ ] Scans filenames for special characters that could cause issues
   [ ] At minimum: < > & \" ' spaces at start/end, control characters
   [ ] Consider: Unicode normalization issues, extremely long filenames
   [ ] Logs a WARNING (not error) for each file with special characters
   [ ] Does NOT modify filenames (per design doc §7.2 and constraint C-006)
   [ ] Warning message includes the filename and which characters are problematic

   Orphaned JSON Detection:
   [ ] Scans DATA_DIR recursively for all .json files
   [ ] For each .json file, checks if corresponding source file exists
   [ ] Source filename is derived by stripping .json suffix
   [ ] Logs a WARNING for each orphaned JSON file
   [ ] Warning message includes the JSON filename and its location
   [ ] Orphaned JSON files are NOT included in the return list of files to process

2. DESIGN ALIGNMENT
   [ ] Scanner is the entry point for file discovery (design doc §2.2)
   [ ] Scanner supports batch mode (returns list of files to process)
   [ ] Scanner does not process files - only discovers them (producer handles processing)
   [ ] Scanner validates filenames but does not modify them (design doc §7.2, constraint C-006)
   [ ] Orphaned JSON detection matches design doc §7.3
   [ ] Return type includes both absolute and relative paths (tasks.md T2.3 completion criteria)

3. CODE QUALITY
   [ ] Scanner class has a clear docstring
   [ ] scan_directory() has a clear docstring with return type documentation
   [ ] All methods have type annotations
   [ ] Uses pathlib.Path for path operations (modern Python)
   [ ] Logging is done via logging.getLogger(__name__)
   [ ] No hardcoded file extensions - use a class constant or config
   [ ] No unnecessary complexity; methods are single-purpose

4. ERROR HANDLING
   [ ] scan_directory() handles PermissionError when walking directories
   [ ] scan_directory() handles OSError for inaccessible directories
   [ ] scan_directory() does not crash on individual file errors
   [ ] Orphaned JSON detection handles edge cases:
   [ ] JSON file with no extension on source (e.g., file.json with no source file)
   [ ] JSON file that is not a metadata file (e.g., config.json in data dir)
   [ ] Special character detection handles edge cases:
   [ ] Empty filenames
   [ ] Filenames with only special characters

5. TESTING READINESS
   [ ] Scanner can be instantiated with a temporary directory for tests
   [ ] scan_directory() returns predictable results for a known directory structure
   [ ] Special character warnings can be captured via caplog
   [ ] Orphaned JSON warnings can be captured via caplog
   [ ] Return type is easily iterable and assertable in tests
   [ ] No filesystem side effects (scanner only reads, never writes)
   [ ] Can test with files that have special characters in names
   [ ] Can test with orphaned JSON files
   [ ] Can test with missing JSON files (should be in results)
   [ ] Can test with outdated JSON files (should be in results)
"""

# =============================================================================
# Cross-Module Integration Checks
# =============================================================================

CROSS_MODULE_CHECKLIST = """
Cross-Module Integration Checks

Dependency Compatibility:
   [ ] config.py does not import from metadata.py, errors.py, or scanner.py
   [ ] metadata.py does not import from config.py, scanner.py (may import from errors.py)
   [ ] errors.py does not import from config.py, scanner.py (may import from metadata.py)
   [ ] scanner.py may import from config.py (for DATA_DIR) and errors.py (for logging)
   [ ] No circular imports exist between any of the four modules

Consistency:
   [ ] All modules use the same logging format and logger naming convention
   [ ] All modules use pathlib.Path for path operations
   [ ] All modules use Python 3.14+ type annotation syntax
   [ ] All modules follow the same docstring style (Google, NumPy, or reST - must be consistent)
   [ ] Exception messages follow a consistent format across modules

pydantic-settings Integration:
   [ ] pydantic-settings is listed as a dependency in pyproject.toml
   [ ] pydantic is listed as a dependency in pyproject.toml
   [ ] Both packages are importable in the environment

Summary Review Table:
   Module        | Functionality | Design | Code Quality | Error Handling | Testability | Overall
   config.py     | [ ]           | [ ]    | [ ]          | [ ]            | [ ]         | [ ]
   metadata.py   | [ ]           | [ ]    | [ ]          | [ ]            | [ ]         | [ ]
   errors.py     | [ ]           | [ ]    | [ ]          | [ ]            | [ ]         | [ ]
   scanner.py    | [ ]           | [ ]    | [ ]          | [ ]            | [ ]         | [ ]

Overall Phase 2 Readiness: [ ] Not Started | [ ] In Progress | [ ] Complete | [ ] Blocked
"""

# =============================================================================
# Master Checklist (combined for programmatic use)
# =============================================================================

PHASE2_MASTER_CHECKLIST = {
    "T2.1": T2_1_CHECKLIST,
    "T2.2": T2_2_CHECKLIST,
    "T2.4": T2_4_CHECKLIST,
    "T2.3": T2_3_CHECKLIST,
    "cross_module": CROSS_MODULE_CHECKLIST,
}

__all__ = [
    "T2_1_CHECKLIST",
    "T2_2_CHECKLIST",
    "T2_3_CHECKLIST",
    "T2_4_CHECKLIST",
    "CROSS_MODULE_CHECKLIST",
    "PHASE2_MASTER_CHECKLIST",
]

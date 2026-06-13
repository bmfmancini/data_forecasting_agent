---
name: python-instructions
applyTo: "**/{backend,frontend,core}/**/*.py"
description: |
  Python-specific guidelines for the Data Forecasting Agent project.
  
  ## Python Version
  - The project uses Python 3.11 as specified in the Dockerfiles
  - All code must be compatible with Python 3.11 features and syntax
  
  ## Code Style and Formatting
  - Follow PEP 8 style guide with 4-space indentation
  - Maximum line length is 88 characters (as per Black formatter defaults)
  - Use meaningful variable and function names in snake_case
  - Class names should use PascalCase
  - Constants should be in UPPER_SNAKE_CASE
  
  ## Type Hints
  - All function parameters and return values must include type hints
  - Use built-in types and typing module (List, Dict, Optional, etc.) where appropriate
  - For complex types, use typing aliases for better readability
  - Use generics where applicable (TypeVar, Generic)
  
  ## Import Organization
  - Imports must be organized in the following order:
    1. Standard library imports
    2. Third-party library imports
    3. Local application imports
  - Within each group, imports should be alphabetized
  - Avoid wildcard imports (from module import *)
  - Use explicit imports rather than implicit relative imports
  
  ## Docstrings
  - Use Google-style docstrings for all public functions, methods, and classes
  - Include parameter descriptions, return value descriptions, and exceptions raised
  - For complex functions, include usage examples in the docstring
  - Module-level docstrings should describe the purpose of the module
  
  ## Error Handling
  - Use specific exception types rather than generic exceptions
  - Create custom exceptions in `backend.exceptions` for domain-specific errors
  - Log errors appropriately using the project's logging configuration
  - Never ignore exceptions silently; handle or propagate them
  
  ## Testing
  - All public functions must have corresponding unit tests
  - Use pytest for testing framework
  - Follow the existing test structure and naming conventions
  - Mock external dependencies and APIs where appropriate
  - Test both success and failure cases
  
  ## Dependencies
  - Add new dependencies to requirements.txt with appropriate version constraints
  - Use uv.txt for uv-based installations in Docker environments
  - Pin dependency versions to ensure reproducible builds
  - Regularly update dependencies and check for security vulnerabilities
  
  ## Performance Considerations
  - Use list comprehensions instead of loops where appropriate
  - Prefer generator expressions for memory efficiency with large datasets
  - Use pandas vectorized operations instead of iterating through DataFrames
  - Cache expensive computations using appropriate caching mechanisms
  - Profile code to identify bottlenecks before optimizing
  
  ## Security
  - Validate all input data before processing
  - Sanitize user inputs to prevent injection attacks
  - Do not hardcode sensitive information; use environment variables
  - Follow the principle of least privilege for file and network access
  - Use secure random number generation for cryptographic purposes
  
  ## Project-Specific Conventions
  - Use the project's logging configuration via `utils.logging_config.get_logger`
  - Follow the established folder structure and module organization
  - Use the existing configuration management in `core.config`
  - Adhere to the architectural boundaries between backend, frontend, and core modules
  - Use the data schemas defined in `utils.schemas` for data validation
---
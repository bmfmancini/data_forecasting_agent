# GitHub Copilot Instructions

## Priority Guidelines

When generating code for this repository:

1. **Version Compatibility**: Always detect and respect the exact versions of languages, frameworks, and libraries used in this project
2. **Context Files**: Prioritize patterns and standards defined in the `.github/copilot` directory
3. **Codebase Patterns**: When context files don't provide specific guidance, scan the codebase for established patterns
4. **Architectural Consistency**: Maintain our **Layered** architectural style and established boundaries
5. **Code Quality**: Prioritize **maintainability, performance, security, and testability** in all generated code

## Technology Version Detection

Before generating code, scan the codebase to identify:

1. **Language Versions**: Detect the exact versions of programming languages in use
   - Examine project files, configuration files, and package managers
   - Look for language‑specific version indicators (e.g., `python_version` in `requirements.txt` or Dockerfiles)
   - Never use language features beyond the detected version

2. **Framework Versions**: Identify the exact versions of all frameworks
   - Check `requirements.txt`, Dockerfiles, and any `pyproject.toml`‑like files
   - Respect version constraints when generating code
   - Never suggest features not available in the detected framework versions

3. **Library Versions**: Note the exact versions of key libraries and dependencies
   - Generate code compatible with these specific versions
   - Never use APIs or features not available in the detected versions

4  ***Tools to always use while implementing code:**
   - `context7` use the context7 mcp server to get the latest documentation and examples for any library or framework in use
   - `black` for Python code formatting
   - `pytest` for testing
   - `mypy` for static type checking

## Context Files

Prioritize the following files in the `.github/copilot` directory (if they exist):

- **architecture.md**: System architecture guidelines
- **tech-stack.md**: Technology versions and framework details
- **coding-standards.md**: Code style and formatting standards
- **folder-structure.md**: Project organization guidelines
- **exemplars.md**: Exemplary code patterns to follow

## Codebase Scanning Instructions

When context files don't provide specific guidance:

1. Identify similar files to the one being modified or created
2. Analyze patterns for:
   - Naming conventions
   - Code organization
   - Error handling
   - Logging approaches
   - Documentation style
   - Testing patterns
3. Follow the most consistent patterns found in the codebase
4. When conflicting patterns exist, prioritize patterns in newer files or files with higher test coverage
5. Never introduce patterns not found in the existing codebase

## Code Quality Standards

### Maintainability
- Write self‑documenting code with clear naming
- Follow the naming and organization conventions evident in the codebase
- Follow established patterns for consistency
- Keep functions focused on single responsibilities
- Limit function complexity and length to match existing patterns

### Performance
- Follow existing patterns for memory and resource management
- Match existing patterns for handling computationally expensive operations
- Follow established patterns for asynchronous operations
- Apply caching consistently with existing patterns
- Optimize according to patterns evident in the codebase

### Security
- Follow existing patterns for input validation
- Apply the same sanitization techniques used in the codebase
- Use parameterized queries matching existing patterns (if any DB access exists)
- Follow established authentication and authorization patterns
- Handle sensitive data according to existing patterns

### Testability
- Follow established patterns for testable code
- Match the dependency injection approaches used in the codebase
- Apply the same patterns for managing dependencies
- Follow established mocking and test double patterns
- Match the testing style used in existing tests

## Documentation Requirements

**Standard**
- Follow the exact documentation format found in the codebase
- Match the docstring style (Google style) used throughout the Python modules
- Document parameters, returns, and exceptions in the same style
- Follow existing patterns for usage examples where present
- Match class‑level documentation style and content

## Testing Approach

### Unit Testing
- Match the exact structure and style of existing unit tests
- Follow the same naming conventions for test classes and methods
- Use the same assertion patterns found in existing tests
- Apply the same mocking approach used in the codebase
- Follow existing patterns for test isolation

### Integration Testing
- Follow the same integration test patterns found in the codebase
- Match existing patterns for test data setup and teardown
- Use the same approach for testing component interactions
- Follow existing patterns for verifying system behavior

## Technology‑Specific Guidelines

### Python Guidelines
- Detect and adhere to the specific Python version in use (as defined in `requirements.txt` / Dockerfile)
- Follow the same import organization found in existing modules (standard library → third‑party → local)
- Match type hinting approaches if used in the codebase
- Apply the same error handling patterns found in existing code
- Follow the same module organization patterns

## Version Control Guidelines

- Follow Semantic Versioning patterns as applied in the codebase
- Match existing patterns for documenting breaking changes
- Follow the same approach for deprecation notices

## General Best Practices

- Follow naming conventions exactly as they appear in existing code
- Match code organization patterns from similar files
- Apply error handling consistent with existing patterns
- Follow the same approach to testing as seen in the codebase
- Match logging patterns from existing code (`utils.logging_config`)
- Use the same approach to configuration as seen in the codebase

## Project‑Specific Guidance

- Scan the codebase thoroughly before generating any code
- Respect existing architectural boundaries without exception
- Match the style and patterns of surrounding code
- When in doubt, prioritize consistency with existing code over external best practices

---

*This file was generated automatically based on the project's existing patterns and the Copilot Instructions Blueprint Generator.*

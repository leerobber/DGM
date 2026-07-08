```markdown
# DGM Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the DGM Python codebase. You'll learn about file naming, import/export styles, commit message conventions, and how to structure and run tests. The repository follows clear, maintainable practices that enable collaborative and scalable Python development.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `data_loader.py`, `model_utils.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import calculate_loss
    from ..models import ModelClass
    ```

### Export Style
- Use **named exports** (explicitly define what is exported in `__all__`).
  - Example:
    ```python
    __all__ = ["MyClass", "my_function"]
    ```

### Commit Messages
- Follow **conventional commits** with the `feat` prefix for new features.
  - Example:
    ```
    feat: add data augmentation to training pipeline
    ```

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new capability or module  
**Command:** `/add-feature`

1. Create a new file using snake_case naming.
2. Write your code, using relative imports for internal modules.
3. Define `__all__` for named exports.
4. Write or update tests in a corresponding `*.test.*` file.
5. Commit your changes with a message starting with `feat:`.
6. Push your branch and open a pull request.

### Writing and Running Tests
**Trigger:** When verifying new or existing code  
**Command:** `/run-tests`

1. Create or update test files matching the pattern `*.test.*`.
2. Write test cases for your functions or classes.
3. Use the chosen (unspecified) test framework to run tests.
   - Example (if using pytest):
     ```
     pytest
     ```
4. Ensure all tests pass before merging.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `data_loader.test.py`).
- The specific test framework is not specified; use the one adopted by your team.
- Place test files alongside or near the code they test.
- Write clear, isolated test cases for each function or class.

  Example test file:
  ```python
  # data_loader.test.py
  from .data_loader import load_data

  def test_load_data_returns_dataframe():
      df = load_data("sample.csv")
      assert df is not None
  ```

## Commands
| Command       | Purpose                                   |
|---------------|-------------------------------------------|
| /add-feature  | Start the workflow for adding a new feature|
| /run-tests    | Run all tests in the repository           |
```

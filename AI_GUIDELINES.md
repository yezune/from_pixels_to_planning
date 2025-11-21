# AI Agent Guidelines: From Pixels to Planning

This document serves as the primary directive for AI agents working on the `from_pixels_to_planning` project. It outlines the mandatory development workflow, testing strategies, and architectural standards that must be followed to maintain project integrity.

## 1. Core Philosophy

The project aims to bridge the gap between raw pixel inputs and high-level planning using Hierarchical Reinforcement Learning (HRL) and Generative Models (RGM).

- **Stability First**: The codebase must remain in a "Green" state (all tests passing) at the end of every task.
- **Rigorous Verification**: No code is considered "done" until it is verified by an automated test.

## 2. Development Workflow: TDD (Test-Driven Development)

All development must follow the TDD cycle:

1. **Red (Write a Failing Test)**:
    - Before implementing a feature or fixing a bug, write a test case that reproduces the requirement or issue.
    - Verify that the test fails.
2. **Green (Make it Pass)**:
    - Write the minimum amount of code necessary to pass the test.
    - Do not over-engineer at this stage.
3. **Refactor (Clean Up)**:
    - Optimize the code structure (e.g., extract methods, introduce base classes) while ensuring tests still pass.

## 3. Testing Strategy

The project relies on a comprehensive testing hierarchy. Agents must use the following commands to verify changes.

### 3.1. Acceptance Testing (The Master Switch)

The acceptance test suite is the ultimate source of truth. It discovers and runs all unit, integration, and notebook tests.

- **Command**:

  ```bash
  python3 -m unittest tests/test_acceptance.py
  ```

- **Rule**: This command must return `OK` before any code is pushed or marked as complete.

### 3.2. Notebook Testing

Jupyter Notebooks are treated as production code. They must be executable from top to bottom without errors.

- **File**: `tests/test_notebooks.py`
- **Mechanism**: Uses `nbconvert` to execute notebooks in a headless environment.
- **Rule**: If you modify a notebook, you must verify it using the notebook test suite.

### 3.3. Unit & Integration Testing

- **Location**: `tests/` directory.
- **Naming Convention**: `test_*.py`.
- **Mocking**: Use `unittest.mock` heavily to isolate components (e.g., mocking Gym environments or PyTorch models) during unit testing.

## 4. Refactoring & Architecture

The project uses a hierarchical class structure to ensure modularity.

### 4.1. Base Classes

- **`BaseTrainer`**: Located in `src/base_trainer.py`. Handles common RL loops, buffer management, and device placement. All specific trainers (e.g., `Phase1Trainer`) must inherit from this.
- **`BaseVAE`**: Located in `src/models/base_vae.py`. Handles common VAE logic like reparameterization and loss computation.

### 4.2. Code Standards

- **Type Hinting**: Use Python type hints for function arguments and return values.
- **Docstrings**: All classes and public methods must have docstrings explaining their purpose, inputs, and outputs.
- **Imports**: Use absolute imports (e.g., `from src.models import ...`) to avoid path issues.

## 5. Instructions for AI Agents

When you are assigned a task, follow this checklist:

1. **Context Analysis**: Read `AI_GUIDELINES.md` (this file) and the current file structure.
2. **Pre-Check**: Run `python3 -m unittest tests/test_acceptance.py` to ensure the starting state is clean.
3. **Plan**: Break down the task into small, testable steps.
4. **Execute (TDD)**:
    - Create/Update a test file in `tests/`.
    - Implement the feature in `src/`.
    - Run the specific test to verify.
5. **Regression Check**: Run the full acceptance suite again to ensure no side effects.
6. **Documentation**: Update `README.md` or notebook explanations if the feature changes user-facing behavior.
7. **Cleanup**: Remove any temporary files or debug prints before finishing.

---
**Note**: If you encounter a `ModuleNotFoundError`, ensure `sys.path` includes the project root, or run tests as modules (`python -m unittest ...`) from the root directory.

# Contributing to TorchQuantum

Thank you for your interest in contributing to TorchQuantum! This document provides guidelines and instructions for contributing.

## Getting Started

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/torchquantum.git
   cd torchquantum
   ```
3. Install in development mode:
   ```bash
   pip install --editable .
   ```

### Branch Strategy

- **main**: Stable release branch
- **dev**: Development branch for integrating features

**Always create pull requests to the `dev` branch, not `main`.**

## Making Contributions

### Pull Request Process

1. Create a new branch from `dev` for your feature or fix:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines below

3. Test your changes thoroughly:
   ```bash
   # Run existing tests
   pytest tests/

   # For new features, add corresponding tests
   ```

4. Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "Brief description of changes"
   ```

5. Push to your fork and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

6. In your PR description:
   - Describe what changes you made and why
   - Reference any related issues (e.g., "Fixes #123")
   - Include test results or examples if applicable

7. Ping a maintainer for review by tagging them in the PR

### Code Style Guidelines

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings to new functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

### Testing Requirements

- **All changes must pass existing tests**
- **New features must include corresponding tests**
- Test edge cases and error conditions
- Ensure backward compatibility when modifying existing functionality

### Documentation

- Update docstrings for any modified functions
- Add examples for new features
- Update README if adding major functionality

## Types of Contributions

### Bug Fixes

1. Check if the bug is already reported in [Issues](https://github.com/mit-han-lab/torchquantum/issues)
2. If not, create a new issue describing the bug
3. Reference the issue in your PR

### New Features

1. Open an issue first to discuss the feature
2. Wait for maintainer feedback before starting implementation
3. Follow the PR process above

### Documentation

- Improvements to README, docstrings, or examples are always welcome
- For major documentation changes, open an issue first

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## Questions?

- Open an issue for questions about contributing
- Tag maintainers for guidance on complex changes

Thank you for helping improve TorchQuantum!

# Contributing to Speite

Thank you for your interest in contributing to Speite! This document provides guidelines for contributing to this offline speech-to-text system.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of speech recognition and audio processing
- Familiarity with FastAPI and PyTorch (for advanced contributions)

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/speite.git
cd speite
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

4. Run tests to ensure everything works:
```bash
pytest tests/ -v
```

## Code Style

### Python Style Guide

- Follow PEP 8 style guidelines
- Use type hints where applicable
- Write comprehensive docstrings (Google style)
- Keep functions focused and single-purpose
- Maximum line length: 100 characters

### Documentation Style

Every module, class, and function should have a docstring:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this error occurs
    """
    pass
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `AudioPreprocessor`)
- **Functions/Methods**: snake_case (e.g., `load_audio`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_DURATION`)
- **Private**: Prefix with underscore (e.g., `_internal_method`)

## Testing

### Writing Tests

- Write tests for all new functionality
- Place tests in the `tests/` directory
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

Example:

```python
def test_audio_validation_rejects_empty_audio():
    """Test that validation fails for empty audio arrays"""
    # Arrange
    preprocessor = AudioPreprocessor()
    empty_audio = np.array([])
    
    # Act & Assert
    with pytest.raises(ValueError, match="empty"):
        preprocessor.validate_audio(empty_audio)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_basic.py

# Run with coverage
pytest --cov=speite tests/

# Run with verbose output
pytest -v
```

## Making Changes

### Branch Naming

- **Feature**: `feature/description` (e.g., `feature/add-speaker-detection`)
- **Bug Fix**: `bugfix/description` (e.g., `bugfix/fix-audio-validation`)
- **Documentation**: `docs/description` (e.g., `docs/update-readme`)
- **Refactor**: `refactor/description` (e.g., `refactor/improve-error-handling`)

### Commit Messages

Use clear, descriptive commit messages:

```
Add support for WAV file validation

- Implement format detection
- Add validation tests
- Update documentation
```

Format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description (if needed)
- Bullet points for changes

### Pull Request Process

1. Create a new branch for your changes
2. Make your changes with appropriate tests
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request with:
   - Clear description of changes
   - Link to any related issues
   - Screenshots (if UI changes)

## Areas for Contribution

### High Priority

1. **Multi-language Support**: Extend beyond English
2. **Performance Optimization**: Improve CPU inference speed
3. **Additional Audio Formats**: Support more formats
4. **Error Handling**: Improve error messages and recovery

### Medium Priority

1. **Batch Processing**: Process multiple files at once
2. **Audio Enhancement**: Pre-processing for noisy audio
3. **Custom Vocabularies**: Domain-specific transcription
4. **WebSocket Support**: Real-time streaming

### Low Priority

1. **UI Dashboard**: Web interface for transcription
2. **Statistics**: Transcription analytics
3. **Export Formats**: Multiple output formats (SRT, VTT, etc.)
4. **Audio Splitting**: Handle very long audio files

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] No unnecessary dependencies added
- [ ] Backward compatibility maintained
- [ ] Comments explain complex logic
- [ ] No hardcoded values (use config)
- [ ] Error handling is appropriate
- [ ] Logging is informative

## Academic Guidelines

### For Academic Research

If using Speite in academic research:

1. **Cite the Project**: Include project reference in your work
2. **Document Changes**: Clearly document any modifications
3. **Share Improvements**: Consider contributing improvements back
4. **Reproducibility**: Ensure your setup is reproducible

### Code for Academic Review

When writing code for academic review:

- Prioritize clarity over cleverness
- Add extensive comments explaining reasoning
- Include references to papers/algorithms used
- Document assumptions and limitations
- Provide examples of usage

## Questions and Support

### Getting Help

- **Issues**: Open an issue on GitHub for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check existing docs first

### Reporting Bugs

When reporting a bug, include:

1. Python version
2. Operating system
3. Steps to reproduce
4. Expected behavior
5. Actual behavior
6. Error messages/logs
7. Sample audio file (if applicable)

## License

By contributing to Speite, you agree that your contributions will be licensed under the same license as the project.

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Accept criticism gracefully
- Prioritize community health

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Unprofessional conduct

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Project README (for significant contributions)
- Release notes

## Thank You!

Your contributions help make Speite better for everyone working in low-connectivity environments. We appreciate your time and effort!

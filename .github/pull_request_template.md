# Pull Request

## Summary
<!-- Brief description of what this PR accomplishes -->

## Type of Change
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🎨 Code style/formatting changes
- [ ] 🧙 Refactoring (no functional changes)
- [ ] ⚙️ Configuration changes
- [ ] 💰 Performance improvements
- [ ] 🔒 Security improvements

## Related Issues
- Closes #(issue number)
- Related to #(issue number)

## Description
<!-- Detailed description of the changes -->

## Changes Made
### Core Changes
- [ ] Modified MCP tools or interfaces
- [ ] Updated session management
- [ ] Changed perspective orchestration logic
- [ ] Modified AORP response formatting
- [ ] Updated LLM backend integration
- [ ] Changed error handling or logging

### Technical Details
**Components Modified:**
- `src/context_switcher_mcp/[component].py` - [description]
- `tests/test_[component].py` - [description]

**Architecture Impact:**
- [ ] No architectural changes
- [ ] Refactored existing components
- [ ] Added new components
- [ ] Changed component interfaces
- [ ] Modified data flow

## Testing
### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All existing tests pass
- [ ] New functionality is covered by tests
- [ ] Test coverage maintained/improved

### Manual Testing
- [ ] Tested with AWS Bedrock backend
- [ ] Tested with LiteLLM backend
- [ ] Tested with Ollama backend
- [ ] Tested multi-perspective analysis
- [ ] Tested session management
- [ ] Tested error scenarios

**Testing Commands Run:**
```bash
# Paste the commands you ran to test this change
pytest tests/
# etc.
```

## Performance Impact
- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance regression (explained below)
- [ ] Performance impact unknown/needs testing

**Performance Notes:**
[If applicable, describe performance implications]

## Security Considerations
- [ ] No security implications
- [ ] Security improvements made
- [ ] Security review needed
- [ ] Sensitive data handling reviewed
- [ ] Input validation updated

## Documentation
- [ ] README updated (if needed)
- [ ] CHANGELOG updated
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] Configuration documentation updated

## Compatibility
### Backward Compatibility
- [ ] Fully backward compatible
- [ ] Backward compatible with migration guide
- [ ] Breaking changes (detailed in migration section)

### MCP Client Compatibility
- [ ] Compatible with all MCP clients
- [ ] Requires specific MCP version (specified below)
- [ ] New optional features (backward compatible)

## Migration Guide (if breaking changes)
**For Users:**
- [ ] No changes required
- [ ] Configuration changes needed
- [ ] Tool interface changes

**Steps to migrate:**
1. [Step 1]
2. [Step 2]

## Screenshots/Examples (if applicable)
**Before:**
```
[Example of old behavior]
```

**After:**
```
[Example of new behavior]
```

## Review Checklist
### Code Quality
- [ ] Code follows project style guidelines (ruff format)
- [ ] Code passes linting (ruff check)
- [ ] Type hints are complete and correct (mypy)
- [ ] Error handling is appropriate
- [ ] Logging is appropriate and consistent

### Testing
- [ ] All tests pass locally
- [ ] CI pipeline passes
- [ ] Edge cases are tested
- [ ] Error scenarios are tested

### Documentation
- [ ] Public APIs are documented
- [ ] Complex logic is commented
- [ ] Breaking changes are documented
- [ ] Examples are provided where helpful

## Additional Notes
<!-- Any additional information that reviewers should know -->

---

**For Reviewers:**
- Please test with your preferred LLM backend
- Pay special attention to [specific areas if applicable]
- This change affects [specific functionality]

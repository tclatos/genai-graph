# GenAI Graph Testing Plan

## Overview

This document outlines the testing strategy for the GenAI Graph project, which started as a rapid prototype and now requires comprehensive testing to prevent regressions and ensure reliability.

## Testing Architecture

### Test Structure
```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── unit_tests/                    # Fast, isolated tests
│   └── core/
│       ├── test_kuzu_type_mapping.py      # Type mapping logic
│       ├── test_embedded_structs.py        # Embedded struct handling
│       └── test_graph_schema.py            # Schema validation (future)
└── integration_tests/             # End-to-end tests
    ├── test_schema_creation.py            # Full database schema creation
    └── test_graph_operations.py           # CRUD operations (future)
```

## Test Categories

### 1. Unit Tests

**Purpose**: Test individual functions and classes in isolation

**Current Coverage**:
- ✅ Type mapping (`_get_kuzu_type`)
  - Basic types (str, int, float)
  - Optional types unwrapping
  - List types
  - Complex type fallbacks
  
- ✅ Embedded struct detection
  - Finding embedded fields in parent classes
  - Optional embedded fields
  - List-based embedded fields
  - Multiple embedded structs per node

**Future Coverage**:
- [ ] Graph schema validation
- [ ] Node configuration deduplication
- [ ] Field path deduction
- [ ] Relationship validation
- [ ] Name generation from callables

### 2. Integration Tests

**Purpose**: Test complete workflows with real database interactions

**Current Coverage**:
- ✅ Schema creation with embedded STRUCTs
  - STRUCT type generation in database
  - Multiple embedded structs per node
  - Various data types in structs (float, int, bool, list)
  - Optional embedded fields

**Future Coverage**:
- [ ] Full CRUD operations
- [ ] Relationship creation and querying
- [ ] Data loading from KV store
- [ ] Graph traversal operations
- [ ] Concurrent access scenarios
- [ ] Schema evolution/migration

### 3. Regression Tests

**Purpose**: Prevent known bugs from reappearing

**Implemented Regressions**:
1. ✅ **Embedded STRUCTs as STRING bug** (2024-12-04)
   - **Issue**: Indentation error caused code after `if not field_name: continue` to be unreachable
   - **Impact**: `financials` and `competition` fields created as STRING instead of STRUCT
   - **Tests**: 
     - `test_schema_with_embedded_structs` - verifies schema processing
     - `test_struct_types_in_database` - verifies database creation
   - **Location**: 
     - Unit: `tests/unit_tests/core/test_embedded_structs.py`
     - Integration: `tests/integration_tests/test_schema_creation.py`

**Future Regression Coverage**:
- [ ] ForwardRef handling
- [ ] Optional type unwrapping
- [ ] Field path deduction edge cases

## Test Execution

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only (fast)
make test-unit

# Run integration tests only
make test-integration

# Run with coverage
uv run pytest tests/ --cov=genai_graph --cov-report=html

# Run specific test file
uv run pytest tests/unit_tests/core/test_embedded_structs.py -v

# Run specific test
uv run pytest tests/unit_tests/core/test_embedded_structs.py::TestEmbeddedStructs::test_find_embedded_field_simple -v
```

### CI/CD Integration

**Recommended GitHub Actions Workflow**:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1
      - run: uv sync
      - run: make lint
      - run: make test
```

## Test Quality Standards

### Code Coverage Goals
- **Unit Tests**: ≥80% coverage of core logic
- **Integration Tests**: ≥60% coverage of public APIs
- **Overall**: ≥70% total coverage

### Test Characteristics
- ✅ **Fast**: Unit tests complete in < 100ms each
- ✅ **Isolated**: No dependency on external services
- ✅ **Deterministic**: Same input always produces same output
- ✅ **Focused**: Each test validates one specific behavior
- ✅ **Clear**: Test names describe what they validate
- ✅ **Maintainable**: Tests use fixtures to reduce duplication

## Fixtures and Test Utilities

### Available Fixtures (conftest.py)

1. **`temp_db_path`**: Provides temporary database file path
   - Auto-cleanup after test
   - Isolated per test

2. **`graph_backend`**: Pre-configured KuzuBackend instance
   - Connected to temporary database
   - Clean state for each test

3. **`sample_pydantic_model`**: Simple Pydantic model for testing
   - Basic field types
   - Optional fields

4. **`nested_pydantic_models`**: Complex nested models
   - Multiple related classes
   - Embedded relationships

## Future Improvements

### Short Term (Next Sprint)
1. Add test coverage for graph operations (create_graph, merge_nodes)
2. Add property-based testing with Hypothesis
3. Add performance benchmarks
4. Set up automated coverage reporting

### Medium Term (Next Month)
1. Add mutation testing to validate test quality
2. Create test data generators for realistic scenarios
3. Add stress tests for large graphs
4. Document test patterns and best practices

### Long Term (Next Quarter)
1. Add contract tests for external integrations
2. Create visual regression tests for HTML output
3. Add security testing (SQL injection, etc.)
4. Establish test data management strategy

## Test Data Management

### Principles
- Use **generated data** (Faker, factories) for most tests
- Use **real examples** only for regression tests
- Keep test data **minimal** but **representative**
- **Anonymize** any real data used in tests

### Test Data Locations
- `tests/fixtures/` - Shared test data files (future)
- `tests/data/` - Test-specific data (future)
- In-test generation using Pydantic models (current)

## Dependencies

### Test Dependencies (from pyproject.toml)
```toml
[dependency-groups.dev]
dev = [
    "ruff>=0.9.1",       # Linting
    "pytest>=8.1.1",     # Testing framework
]
```

### Additional Testing Tools (Future)
- `pytest-cov` - Code coverage
- `pytest-xdist` - Parallel test execution
- `pytest-benchmark` - Performance testing
- `hypothesis` - Property-based testing
- `faker` - Test data generation

## Metrics and Reporting

### Key Metrics to Track
1. **Test Count**: Current = 15 unit + 3 integration
2. **Code Coverage**: Target ≥70%
3. **Test Execution Time**: Target < 10s for full suite
4. **Flaky Tests**: Target = 0
5. **Test Maintenance Time**: Monitor and reduce

### Current Status
- ✅ Test infrastructure set up
- ✅ Core regression tests implemented
- ✅ Make targets configured
- ⏳ Coverage tracking not yet enabled
- ⏳ CI/CD not yet configured

## Contributing Guidelines

### Writing New Tests
1. **Identify what to test**: New feature or bug fix
2. **Choose test type**: Unit vs integration
3. **Write test first** (TDD when possible)
4. **Use descriptive names**: `test_[what]_[condition]_[expected]`
5. **Add docstring**: Explain why the test exists
6. **Use fixtures**: Leverage existing test infrastructure
7. **Keep it simple**: One assertion per test when possible
8. **Document regressions**: Add comments explaining historical bugs

### Test Review Checklist
- [ ] Test names are descriptive
- [ ] Tests are properly categorized (unit/integration)
- [ ] Tests use appropriate fixtures
- [ ] Tests clean up after themselves
- [ ] Tests run in isolation (no order dependency)
- [ ] Regression tests document the original bug
- [ ] Performance impact is acceptable

## Known Limitations

1. **Database isolation**: Tests use file-based temp DB (could be slower)
2. **External dependencies**: Some features require external services (not yet mocked)
3. **Test data**: Limited real-world test scenarios
4. **Coverage gaps**: Frontend/CLI testing not yet implemented

## References

- [pytest documentation](https://docs.pytest.org/)
- [Pydantic testing best practices](https://docs.pydantic.dev/latest/concepts/testing/)
- [Test-Driven Development principles](https://martinfowler.com/bliki/TestDrivenDevelopment.html)

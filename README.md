# Type Inference for Generic Function Return Types

A sophisticated type inference system for Python generic functions that infers concrete return types from runtime arguments using formal unification algorithms.

## Overview

This library solves the problem of inferring concrete types for TypeVars in generic function return types based on the actual arguments passed at runtime.

```python
from typing import TypeVar, List
from unification_type_inference import infer_return_type_unified

A = TypeVar('A')

def head(items: List[A]) -> A:
    """Get first item from list."""
    return items[0]

# Infer that return type is int
result_type = infer_return_type_unified(head, [1, 2, 3])
assert result_type is int

# Infer that return type is str  
result_type = infer_return_type_unified(head, ['hello', 'world'])
assert result_type is str
```

## Features

- ✅ **Unification-based algorithm**: Formal type unification with constraint solving
- ✅ **Comprehensive generic support**: Works with dataclasses, Pydantic models, and custom generics
- ✅ **Union type formation**: Automatically creates unions for mixed-type containers
- ✅ **Nested structure handling**: Handles deeply nested generic structures
- ✅ **TypeVar bounds/constraints**: Enforces TypeVar bounds and explicit constraints
- ✅ **Type overrides**: Supports manual type overrides for edge cases
- ✅ **Variance awareness**: Handles covariant, contravariant, and invariant positions

## Installation

```bash
# Using uv
uv pip install -e .

# Or using pip
pip install -e .
```

## Usage

### Basic Example

```python
from typing import TypeVar, List, Dict, Tuple
from unification_type_inference import infer_return_type_unified

A = TypeVar('A')
B = TypeVar('B')

# Simple list inference
def merge_lists(a: List[A], b: List[A]) -> List[A]:
    return a + b

result_type = infer_return_type_unified(merge_lists, [1, 2], [3, 4])
# Returns: list[int]

# Dict with multiple TypeVars
def invert_dict(d: Dict[A, B]) -> Dict[B, A]:
    return {v: k for k, v in d.items()}

result_type = infer_return_type_unified(invert_dict, {1: 'a', 2: 'b'})
# Returns: dict[str, int]
```

### Mixed Type Containers

```python
def process_mixed(items: List[A]) -> A:
    return items[0]

# Automatically creates union types
result_type = infer_return_type_unified(process_mixed, [1, 'hello', 3.14])
# Returns: int | str | float
```

### Generic Classes

```python
from dataclasses import dataclass
from pydantic import BaseModel
import typing

@dataclass
class Wrap(typing.Generic[A]):
    value: A

def unwrap(w: Wrap[A]) -> A:
    return w.value

result_type = infer_return_type_unified(unwrap, Wrap[int](42))
# Returns: int

# Works with Pydantic too
class Box(BaseModel, typing.Generic[A]):
    item: A

def unbox(boxes: List[Box[A]]) -> List[A]:
    return [b.item for b in boxes]

result_type = infer_return_type_unified(unbox, [Box[str](item='hello')])
# Returns: list[str]
```

### Type Overrides

```python
# For empty containers, use type overrides
def head(items: List[A]) -> A:
    return items[0]

result_type = infer_return_type_unified(head, [], type_overrides={A: int})
# Returns: int
```

### Complex Nested Structures

```python
def complex_nested(data: Dict[A, List[B]]) -> Tuple[A, B]:
    pass

result_type = infer_return_type_unified(
    complex_nested, 
    {'key': [1, 2, 3]}
)
# Returns: tuple[str, int]
```

## API Reference

### Main Function

```python
infer_return_type_unified(
    fn: callable,
    *args,
    type_overrides: Optional[Dict[TypeVar, type]] = None,
    **kwargs
) -> type
```

**Parameters**:
- `fn`: Function with generic type annotations
- `*args`: Positional arguments to the function
- `type_overrides`: Optional dict mapping TypeVars to concrete types
- `**kwargs`: Keyword arguments to the function

**Returns**: Concrete type for the return type annotation

**Raises**: `TypeInferenceError` if types cannot be inferred

## Algorithm

The unification-based algorithm works by:

1. **Constraint Collection**: Extracts type constraints from function parameters
2. **Constraint Solving**: Solves the constraint system using unification
3. **Variance Handling**: Respects covariant/contravariant/invariant positions
4. **Union Formation**: Creates unions when multiple types are valid
5. **Bounds Checking**: Validates TypeVar bounds and constraints
6. **Substitution**: Applies the solution to the return type annotation

## Current Limitations

Some features are planned but not yet implemented (see skipped tests):

- ⚠️ **Conflicting invariant constraints**: Currently fails, should create unions
- ⚠️ **None filtering**: Includes None in unions for Optional[A]
- ⚠️ **Complex union structures**: Some patterns like `Union[A, List[A]]` fail
- ⚠️ **Callable type inference**: Cannot infer from function signatures yet

See `test_infer_return_type_unified.py` for tests marked with `@pytest.mark.skip`.

## Testing

Run the test suite:

```bash
# Run all tests
pytest test_infer_return_type_unified.py -v

# Run only passing tests (skip known limitations)
pytest test_infer_return_type_unified.py -v -k "not skip"

# Run specific test
pytest test_infer_return_type_unified.py::test_basic_containers -v
```

**Test Statistics**:
- 50 passing tests (core functionality)
- 19 skipped tests (documented limitations/TODOs)
- Total: 69 comprehensive tests

## Project Structure

```
infer_return_type/
├── unification_type_inference.py     # Main implementation (1,117 lines)
├── generic_utils.py                   # Generic type utilities (553 lines)
├── test_infer_return_type_unified.py # Test suite (69 tests: 50 passing, 19 skipped)
├── test_generic_utils.py              # Utility tests (55 tests passing)
├── README.md                          # This file
├── pyproject.toml                     # Project configuration
├── uv.lock                            # Dependency lock file
└── docs/                              # Documentation
    ├── CLEANUP_PLAN.md                       # Cleanup planning
    ├── CLEANUP_SUMMARY.md                    # Cleanup results
    ├── FINAL_VERIFICATION_REPORT.md          # Complete verification
    ├── IMPLEMENTATION_COMPARISON_SUMMARY.md  # Historical comparison
    ├── MIGRATION_TO_UNIFICATION_GUIDE.md     # Migration roadmap
    ├── TEST_MIGRATION_VERIFICATION.md        # Test coverage verification
    ├── UNIFICATION_GAPS_ANALYSIS.md          # Known gaps and fixes needed
    └── UNIFICATION_TEST_SUMMARY.md           # Test documentation
```

## Contributing

See `docs/MIGRATION_TO_UNIFICATION_GUIDE.md` for the roadmap to address current limitations.

Priority fixes needed:
1. Conflicting TypeVar bindings should create unions
2. None filtering in Optional[A]  
3. Complex union structure handling

## Development History

This project evolved through three implementations:
1. **Original**: Simple direct binding (removed)
2. **CSP**: Constraint satisfaction problem solver (removed)
3. **Unification**: Current - formal unification algorithm

See `IMPLEMENTATION_COMPARISON_SUMMARY.md` for detailed comparison.

## License

MIT License (or your preferred license)

## Related Work

- [PEP 484](https://www.python.org/dev/peps/pep-0484/) - Type Hints
- [PEP 544](https://www.python.org/dev/peps/pep-0544/) - Protocols
- [Python typing module](https://docs.python.org/3/library/typing.html)
- [Pydantic](https://docs.pydantic.dev/) - Generic model support
- [mypy](https://mypy.readthedocs.io/) - Static type checker

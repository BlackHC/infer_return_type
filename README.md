# Type Inference for Generic Function Return Types

A sophisticated type inference system for Python generic functions that infers concrete return types from runtime arguments using formal unification algorithms. This library solves the fundamental problem of determining what `TypeVar` parameters should be bound to based on actual function arguments.

## Overview

Python's type system allows you to write generic functions with `TypeVar` parameters, but it doesn't provide runtime type inference. This library bridges that gap by analyzing function signatures and runtime arguments to determine concrete return types.

### The Problem

```python
from typing import TypeVar, List

A = TypeVar('A')

def head(items: List[A]) -> A:
    """Get first item from list."""
    return items[0]

# What should the return type be?
result = head([1, 2, 3])  # Should be int
result = head(['a', 'b'])  # Should be str
```

### The Solution

```python
from infer_return_type import infer_return_type

# Infer that return type is int
result_type = infer_return_type(head, [1, 2, 3])
assert result_type is int

# Infer that return type is str  
result_type = infer_return_type(head, ['hello', 'world'])
assert result_type is str
```

## Features

- ✅ **Formal Unification Algorithm**: Implements constraint-based type unification with variance awareness
- ✅ **Comprehensive Generic Support**: Works with built-ins, dataclasses, Pydantic models, and custom generics
- ✅ **Automatic Union Formation**: Creates unions for mixed-type containers (e.g., `int | str`)
- ✅ **Deep Structure Handling**: Supports arbitrarily nested generic structures
- ✅ **TypeVar Validation**: Enforces bounds and constraints with detailed error messages
- ✅ **Type Overrides**: Manual type specification for edge cases (empty containers, etc.)
- ✅ **Variance Awareness**: Handles covariant, contravariant, and invariant positions correctly
- ✅ **Rich Error Messages**: Detailed diagnostics for unification failures and type errors

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .

# For development
git clone <repository-url>
cd infer_return_type
uv sync  # Install dependencies
```

## Usage

### Basic Examples

```python
from typing import TypeVar, List, Dict, Tuple
from infer_return_type import infer_return_type

A = TypeVar('A')
B = TypeVar('B')

# Simple list inference
def merge_lists(a: List[A], b: List[A]) -> List[A]:
    return a + b

result_type = infer_return_type(merge_lists, [1, 2], [3, 4])
print(result_type)  # <class 'list[int]'>

# Dict with multiple TypeVars
def invert_dict(d: Dict[A, B]) -> Dict[B, A]:
    return {v: k for k, v in d.items()}

result_type = infer_return_type(invert_dict, {1: 'a', 2: 'b'})
print(result_type)  # <class 'dict[str, int]'>

# Tuple inference
def pair_values(x: A, y: B) -> Tuple[A, B]:
    return (x, y)

result_type = infer_return_type(pair_values, 42, "hello")
print(result_type)  # <class 'tuple[int, str]'>
```

### Mixed Type Containers

The system automatically creates union types when containers have mixed element types:

```python
def process_mixed(items: List[A]) -> A:
    return items[0]

# Automatically creates union types
result_type = infer_return_type(process_mixed, [1, 'hello', 3.14])
print(result_type)  # int | str | float

# Works with sets too
def process_set(items: Set[A]) -> A:
    return next(iter(items))

result_type = infer_return_type(process_set, {1, 'hello', 3.14})
print(result_type)  # int | str | float
```

### Generic Classes

Works seamlessly with dataclasses and Pydantic models:

```python
from dataclasses import dataclass
from pydantic import BaseModel
import typing

# Dataclass example
@dataclass
class Wrap(typing.Generic[A]):
    value: A

def unwrap(w: Wrap[A]) -> A:
    return w.value

result_type = infer_return_type(unwrap, Wrap[int](42))
print(result_type)  # <class 'int'>

# Pydantic example
class Box(BaseModel, typing.Generic[A]):
    item: A

def unbox(boxes: List[Box[A]]) -> List[A]:
    return [b.item for b in boxes]

result_type = infer_return_type(unbox, [Box[str](item='hello')])
print(result_type)  # <class 'list[str]'>

# Nested generic structures
def extract_nested(data: Dict[str, List[Box[A]]]) -> List[A]:
    result = []
    for boxes in data.values():
        result.extend([box.item for box in boxes])
    return result

result_type = infer_return_type(
    extract_nested, 
    {"key": [Box[int](item=42), Box[int](item=24)]}
)
print(result_type)  # <class 'list[int]'>
```

### Type Overrides

For edge cases like empty containers, you can provide manual type overrides:

```python
# Empty containers - no type information available
def head(items: List[A]) -> A:
    return items[0]

# Use type overrides to specify the expected type
result_type = infer_return_type(head, [], type_overrides={A: int})
print(result_type)  # <class 'int'>

# Multiple TypeVars
def extract_keys_values(d: Dict[A, B]) -> Tuple[List[A], List[B]]:
    return list(d.keys()), list(d.values())

# Override both TypeVars
result_type = infer_return_type(
    extract_keys_values, 
    {}, 
    type_overrides={A: str, B: int}
)
print(result_type)  # <class 'tuple[list[str], list[int]]'>
```

### Complex Nested Structures

Handles arbitrarily deep and complex generic structures:

```python
def complex_nested(data: Dict[A, List[B]]) -> Tuple[A, B]:
    key = next(iter(data.keys()))
    value = data[key][0]
    return key, value

result_type = infer_return_type(
    complex_nested, 
    {'key': [1, 2, 3]}
)
print(result_type)  # <class 'tuple[str, int]'>

# Multi-level nesting
def deeply_nested(data: Dict[A, List[Dict[B, List[C]]]]) -> Tuple[A, B, C]:
    key = next(iter(data.keys()))
    inner_dict = data[key][0]
    inner_key = next(iter(inner_dict.keys()))
    inner_value = inner_dict[inner_key][0]
    return key, inner_key, inner_value

result_type = infer_return_type(
    deeply_nested,
    {'outer': [{'inner': [42]}]}
)
print(result_type)  # <class 'tuple[str, str, int]'>
```

## API Reference

### Main Function

```python
infer_return_type(
    fn: callable,
    *args,
    type_overrides: Optional[Dict[TypeVar, type]] = None,
    **kwargs
) -> type
```

**Parameters**:
- `fn`: Function with generic type annotations (must have return type annotation)
- `*args`: Positional arguments to the function
- `type_overrides`: Optional dict mapping TypeVars to concrete types for edge cases
- `**kwargs`: Keyword arguments to the function

**Returns**: Concrete type for the return type annotation

**Raises**: 
- `ValueError`: If function lacks return type annotation
- `TypeInferenceError`: If types cannot be inferred

### Error Handling

The system provides detailed error messages for common issues:

```python
# Missing return annotation
def no_return_annotation(x: int):
    return x

try:
    infer_return_type(no_return_annotation, 42)
except ValueError as e:
    print(e)  # "Function must have return type annotation"

# Type conflicts
def conflicting_types(x: A, y: A) -> A:
    return x

try:
    infer_return_type(conflicting_types, 1, "hello")
except TypeInferenceError as e:
    print(e)  # Detailed unification error message
```

## Algorithm

The unification-based algorithm implements formal type inference through constraint solving:

### 1. Constraint Collection
Extracts type constraints by analyzing the structural relationship between annotations and runtime values:
- Direct TypeVar bindings: `A` in annotation matches `int` in value
- Container element constraints: `List[A]` with `[1, 2, 3]` creates `A ~ int`
- Nested structure analysis: Recursively processes complex generic types

### 2. Constraint Solving
Solves the constraint system using unification with variance awareness:
- **Invariant constraints**: Must be exactly the same type
- **Covariant constraints**: Allow union formation for mixed types
- **Override constraints**: Take precedence over other constraints

### 3. Union Formation
Automatically creates unions when multiple types are valid:
- Mixed container elements: `[1, 'hello']` → `int | str`
- Conflicting constraints: `A ~ int` and `A ~ str` → `A ~ int | str`

### 4. Bounds and Constraints Validation
Enforces TypeVar bounds and explicit constraints:
- Bound checking: `TypeVar('T', bound=int)` ensures `T` is a subtype of `int`
- Constraint validation: `TypeVar('T', int, str)` ensures `T` is exactly `int` or `str`

### 5. Type Substitution
Applies solved TypeVar bindings to return type annotations to produce concrete types.

## Current Limitations

Some advanced features are not yet implemented (see skipped tests):

- ⚠️ **Callable Type Inference**: Cannot infer from function signatures yet
- ⚠️ **ForwardRef Handling**: String-based forward references not fully supported
- ⚠️ **typing.Any Support**: The `Any` type is not supported
- ⚠️ **PEP Features**: Literal types (PEP 586), Final annotations (PEP 591), and Annotated types (PEP 593) are not supported

See `test_infer_return_type.py` for tests marked with `@pytest.mark.skip` for detailed examples of current limitations.

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest test_infer_return_type.py -v

# Run only passing tests (skip known limitations)
uv run pytest test_infer_return_type.py -v -k "not skip"

# Run specific test
uv run pytest test_infer_return_type.py::test_basic_containers -v

# Run with coverage
uv run pytest test_infer_return_type.py --cov=. --cov-report=html

# Run all test files
uv run pytest -v
```

**Test Statistics**:
- **50+ passing tests** covering core functionality
- **5 skipped tests** documenting current limitations  
- **Total: 55+ comprehensive tests** with detailed examples

**Test Categories**:
- Basic container types (list, dict, tuple, set)
- Generic classes (dataclasses, Pydantic models)
- Union type handling
- TypeVar bounds and constraints
- Complex nested structures
- Error handling and edge cases

## Project Structure

```
infer_return_type/
├── infer_return_type.py              # Main implementation (unification algorithm)
├── generic_utils.py                  # Generic type utilities (structural extraction)
├── test_infer_return_type.py         # Main test suite (55+ tests: 50+ passing, 5 skipped)
├── test_generic_utils.py             # Utility tests (55 tests passing)
├── test_optimization_pydantic_models.py  # Complex Pydantic model tests
├── README.md                         # This file
├── pyproject.toml                    # Project configuration
├── uv.lock                           # Dependency lock file
└── docs/                             # Documentation
    ├── CLEANUP_PLAN.md               # Cleanup planning
    ├── CLEANUP_SUMMARY.md            # Cleanup results
    ├── FINAL_VERIFICATION_REPORT.md  # Complete verification
    ├── IMPLEMENTATION_COMPARISON_SUMMARY.md  # Historical comparison
    ├── MIGRATION_TO_UNIFICATION_GUIDE.md     # Migration roadmap
    ├── TEST_MIGRATION_VERIFICATION.md        # Test coverage verification
    ├── UNIFICATION_GAPS_ANALYSIS.md          # Known gaps and fixes needed
    └── UNIFICATION_TEST_SUMMARY.md           # Test documentation
```

**Key Files**:
- `infer_return_type.py`: Core unification algorithm and type inference engine
- `generic_utils.py`: Structural type extraction utilities for different type systems
- `test_infer_return_type.py`: Comprehensive test suite with examples and edge cases

## Contributing

We welcome contributions! See `docs/MIGRATION_TO_UNIFICATION_GUIDE.md` for the roadmap to address current limitations.

**Priority fixes needed**:
1. **Callable type inference**: Infer from function signatures
2. **ForwardRef handling**: Improve string-based forward reference resolution
3. **typing.Any support**: Add support for the `Any` type
4. **PEP features**: Add support for Literal, Final, and Annotated types

**Development Guidelines**:
- Follow the existing code style and patterns
- Add comprehensive tests for new features
- Update documentation for API changes
- Use `uv` for dependency management

## Development History

This project evolved through three major implementations:

1. **Original**: Simple direct binding approach (removed)
2. **CSP**: Constraint satisfaction problem solver (removed)  
3. **Unification**: Current implementation using formal unification algorithms

See `docs/IMPLEMENTATION_COMPARISON_SUMMARY.md` for detailed comparison of approaches.

## License

MIT License

## Related Work

- [PEP 484](https://www.python.org/dev/peps/pep-0484/) - Type Hints
- [PEP 544](https://www.python.org/dev/peps/pep-0544/) - Protocols  
- [PEP 585](https://www.python.org/dev/peps/pep-0585/) - Type Hint Generics In Standard Collections
- [Python typing module](https://docs.python.org/3/library/typing.html)
- [Pydantic](https://docs.pydantic.dev/) - Generic model support
- [mypy](https://mypy.readthedocs.io/) - Static type checker
- [pyright](https://github.com/microsoft/pyright) - Type checker for Python

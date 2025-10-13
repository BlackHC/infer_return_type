# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python library that implements a sophisticated type inference system for generic functions, using formal unification algorithms to infer concrete return types from runtime arguments. The project uses a constraint-based unification approach to solve TypeVar bindings.

## Key Commands

### Testing
```bash
# Run all tests
uv run pytest test_infer_return_type_unified.py -v

# Run only passing tests (skip known limitations)
uv run pytest test_infer_return_type_unified.py -v -k "not skip"

# Run specific test
uv run pytest test_infer_return_type_unified.py::test_basic_containers -v

# Run with coverage
uv run pytest test_infer_return_type_unified.py --cov=. --cov-report=html

# Run all test files
uv run pytest -v

# Run tests for generic_utils
uv run pytest test_generic_utils.py -v
```

### Installation
```bash
# Install dependencies (uv handles this automatically)
uv sync
```

## Architecture

### Core Components

**1. Unification Engine (`unification_type_inference.py`)**
- Main entry point: `infer_return_type_unified(fn, *args, type_overrides=None, **kwargs) -> type`
- Core algorithm stages:
  1. **Constraint Collection**: `_collect_constraints()` - Recursively extracts type constraints from function parameters by matching annotations with values
  2. **Constraint Solving**: `_solve_constraints()` - Solves the constraint system using unification with variance awareness
  3. **Substitution**: Applies solved TypeVar bindings to return type annotation

**2. Generic Utilities (`generic_utils.py`)**
- Provides a unified interface for extracting type information across different generic type systems
- Key abstraction: `GenericInfo` - Structural representation of generic types with:
  - `origin`: Base generic type (e.g., `list` for `list[int]`)
  - `concrete_args`: Type arguments as nested `GenericInfo` objects
  - `type_params`: TypeVars extracted from arguments
  - `resolved_type`: Fully materialized type (cached property)
- Extractor pattern for different type systems:
  - `BuiltinExtractor`: list, dict, tuple, set
  - `PydanticExtractor`: Pydantic BaseModel generics
  - `DataclassExtractor`: dataclass generics with inheritance support
  - `UnionExtractor`: Union and Optional types

**3. Key Classes**

**Constraint** (`unification_type_inference.py:41-55`)
- Represents `TypeVar ~ concrete_type` with variance (COVARIANT, CONTRAVARIANT, INVARIANT)
- `is_override` flag for explicit type overrides from user

**Substitution** (`unification_type_inference.py:58-89`)
- Maps TypeVars to concrete types
- Supports composition for chaining substitutions
- `apply()` method substitutes TypeVars in annotations

**UnificationEngine** (`unification_type_inference.py:92-681`)
- Core constraint collection and solving logic
- Special handling for:
  - Union types: `_handle_union_annotation()` tries alternatives and picks best match
  - Mixed-type containers: Creates union types via covariant constraints
  - Nested structures: Recursive constraint collection
  - TypeVar distribution: `_try_distribute_union_types()` for patterns like `Set[Union[A, B]]`

### Type Inference Algorithm

**Variance Handling**:
- **COVARIANT**: Allows union formation (e.g., `List[A]` with mixed types → `A = int | str`)
- **INVARIANT**: Requires exact matches (e.g., `Dict[A, B]` keys), but multiple independent sources create unions
- **CONTRAVARIANT**: Currently limited support (Callable parameter types)

**Constraint Resolution Strategy**:
1. Group constraints by TypeVar
2. First pass: Resolve unambiguous constraints (single type or all same type)
3. Second pass: Resolve remaining constraints with union formation
4. Check TypeVar bounds and explicit constraints
5. Apply overrides (highest priority)

**Key Design Decisions**:
- TypeVar bounds follow PEP 484 strictly: `int` is NOT a subtype of `float` in the type system
- Constrained TypeVars (e.g., `TypeVar('T', int, str)`) must resolve to ONE of the specified types, not a union
- Union formation happens for covariant positions and multiple independent invariant sources
- None values are included in unions (not filtered out from Optional types)

### Extractor Pattern Details

**get_annotation_value_pairs()** - Critical method that bridges annotations with values:
- Returns `List[Tuple[GenericInfo, Any]]` mapping annotation structure to actual values
- For `list[A]` with `[1, 2, 3]`: Returns `[(GenericInfo(origin=A), 1), (GenericInfo(origin=A), 2), (GenericInfo(origin=A), 3)]`
- For `dict[A, B]` with `{"key": 42}`: Returns `[(GenericInfo(origin=A), "key"), (GenericInfo(origin=B), 42)]`
- For dataclasses/Pydantic: Returns field-value pairs with TypeVar substitution

**Inheritance Handling** (`DataclassExtractor`):
- Only extracts fields defined in the annotation class, not inherited fields (prevents TypeVar shadowing)
- `_build_inheritance_aware_substitution()` handles swapped TypeVars in inheritance chains
- Example: `class HasB(HasA[B, A], Generic[A, B])` - TypeVars are swapped through inheritance

**Pydantic Specialization**:
- Pydantic automatically specializes field annotations in parameterized classes
- `Level3[bool]` has fields already specialized to `bool`, no manual substitution needed
- Detected via `__pydantic_generic_metadata__` attribute

### Known Limitations

**Documented in skipped tests** (`test_infer_return_type_unified.py`):
1. ⚠️ **None filtering**: None values are included in unions for Optional[A] instead of being filtered out
2. ⚠️ **Complex union structures**: Patterns like `Union[A, List[A], Dict[str, A]]` may fail with conflicting type assignments
3. ⚠️ **Callable type inference**: Cannot extract types from function signatures yet (requires deeper signature inspection)

**Test Statistics**: 50 passing tests (core functionality), 19 skipped tests (documented limitations), Total: 69 comprehensive tests

## Development Notes

### Adding New Generic Type Support

To support a new generic type system:
1. Create a new `GenericExtractor` subclass in `generic_utils.py`
2. Implement `can_handle_annotation()`, `can_handle_instance()`, `extract_from_annotation()`, `extract_from_instance()`
3. Implement `get_annotation_value_pairs()` for constraint extraction
4. Implement `_get_original_type_parameters()` to extract TypeVars from class definitions
5. Add extractor to `GenericTypeUtils.__init__()` extractors list

### Testing Patterns

Tests are organized into categories:
- **Basic single TypeVar** (test_basic_containers, test_optional_and_union)
- **Multi-TypeVar interactions** (test_complex_nested_dict_multiple_typevars)
- **Deep/complex nesting** (test_consolidated_nested_generics)
- **Advanced features** (test_advanced_inheritance_and_specialization)
- **Edge cases** (test_empty_container_inference_limitations)
- **Coverage improvements** (test_constraint_and_substitution_internals)

Use descriptive test names and docstrings explaining the pattern being tested.

### Debugging Type Inference

1. Check constraint collection: Add `print(constraints)` in `_collect_constraints()`
2. Check constraint solving: Add `print(substitution)` after `_solve_constraints()`
3. Use `get_generic_info()` to inspect type structure: `get_generic_info(annotation).concrete_args`
4. Verify extractors: Test `get_annotation_value_pairs()` directly with your annotation/instance

### Common Pitfalls

- **ForwardRef handling**: Use `get_type_hints()` with proper `globalns`/`localns` for resolution
- **TypeVar identity**: Same TypeVar name doesn't mean same TypeVar object (check with `is`)
- **Pydantic same-TypeVar optimization**: `Box[A].resolved_type` returns `Box` (loses TypeVar), use `GenericInfo` structure matching instead
- **Inheritance substitution**: Parent class TypeVars may be swapped/reordered in child classes

## Project History

This is the third implementation iteration:
1. **Original**: Simple direct binding (removed)
2. **CSP**: Constraint satisfaction problem solver (removed)
3. **Unification**: Current implementation using formal unification algorithm

Documentation in `docs/` contains migration guides and comparison summaries.
- When making changes in multiple files always make sure to also write or update unit tests for each changed module.
- Try to add abstractions and deduplicate code. Think about the big picture and avoid special casing as much as possible. Ensure that there is enough test coverage for edge cases and consider Python's type system and features in the latest versions.
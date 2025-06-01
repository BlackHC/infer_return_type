# Unification-Based Type Inference Design

## Overview

This document describes a unification-based type inference system that addresses the limitations of the current annotation-driven approach. The new system treats type inference as a **constraint satisfaction problem** where we unify annotation structures with concrete value types.

## Key Problems with Current System

### 1. Conflict-Based Failures
The current system immediately fails when TypeVar constraints conflict:
```python
def process_nested_mixed(data: List[List[A]]) -> A: ...
process_nested_mixed([[1, 2], ["a", "b"]])  # FAILS with TypeInferenceError
```

### 2. Special-Case Proliferation  
Different code paths for each type system:
- `_extract_from_custom_generic()` for general cases
- `_extract_from_dataclass_fields_fallback()` for dataclasses
- `_extract_from_instance_attributes()` for heuristics
- Separate handling for Pydantic metadata

### 3. Limited Variance Support
No handling of covariance/contravariance relationships.

### 4. No Constraint Enforcement
TypeVar bounds and constraints are not properly validated.

### 5. Brittle Architecture
Adding support for new generic type systems requires modifying core logic.

## Unification-Based Solution

### Core Concepts

#### 1. Constraint Collection
Instead of immediately binding TypeVars, we collect **constraints** that express relationships:
```python
class Constraint:
    typevar: TypeVar
    concrete_type: type  
    variance: Variance  # COVARIANT, CONTRAVARIANT, INVARIANT
```

#### 2. Variance-Aware Resolution
Constraints are resolved differently based on variance:
- **Covariant**: Form unions when conflicts arise (`List[A]` → `A = int | str`)
- **Contravariant**: Find common supertypes 
- **Invariant**: Require exact matches or fail

#### 3. Pluggable Type Extractors
Clean interface for different generic type systems:
```python
class TypeExtractor(ABC):
    def can_handle(self, annotation: Any, instance: Any) -> bool: ...
    def extract_type_params(self, annotation: Any) -> List[TypeVar]: ...
    def extract_concrete_types(self, instance: Any) -> List[type]: ...
    def get_variance(self, annotation: Any, param_index: int) -> Variance: ...
```

#### 4. Constraint Solving
After collecting all constraints, solve the system:
1. Group constraints by TypeVar
2. Resolve conflicts based on variance
3. Check TypeVar bounds and constraints
4. Produce final substitution

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   UnificationEngine                        │
├─────────────────────────────────────────────────────────────┤
│  1. Collect constraints from annotation/value pairs        │
│  2. Solve constraint system                                │  
│  3. Apply substitution to return annotation                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 Type Extractors                            │
├─────────────────────────────────────────────────────────────┤
│  • PydanticExtractor: __pydantic_generic_metadata__        │
│  • DataclassExtractor: __orig_class__ + get_args()         │
│  • BuiltinExtractor: list, dict, tuple, set inference      │
│  • CustomExtractor: Extensible for new type systems       │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              Constraint Solver                             │
├─────────────────────────────────────────────────────────────┤
│  • Variance-aware conflict resolution                      │
│  • TypeVar bounds/constraint checking                      │
│  • Union formation for covariant conflicts                 │
│  • Supertype inference for contravariant conflicts         │
└─────────────────────────────────────────────────────────────┘
```

## Key Improvements

### 1. Union Formation Instead of Conflicts

**Before:**
```python
# FAILS: Conflicting types for A: int vs str
process_nested_mixed([[1, 2], ["a", "b"]])
```

**After:**  
```python
# SUCCESS: Returns int | str  
process_nested_mixed([[1, 2], ["a", "b"]])  # → str | int
```

### 2. Variance-Aware Handling

```python
# Covariant: List[A] allows union formation
def process_list(items: List[A]) -> A: ...
process_list([1, "hello"])  # → int | str ✓

# Invariant: Dict keys require exact matches  
def process_dict_keys(d: Dict[A, str]) -> A: ...
process_dict_keys({1: "a", "b": "c"})  # → int | str (special handling)
```

### 3. TypeVar Bounds and Constraints

```python
T_BOUNDED = TypeVar('T_BOUNDED', bound=int)
T_CONSTRAINED = TypeVar('T_CONSTRAINED', int, str)

def process_bounded(x: T_BOUNDED) -> T_BOUNDED: ...
process_bounded(True)    # ✓ bool ≤ int
process_bounded("hi")    # ✗ str ≰ int

def process_constrained(x: T_CONSTRAINED) -> T_CONSTRAINED: ...
process_constrained(42)     # ✓ int ∈ {int, str}  
process_constrained(3.14)   # ✗ float ∉ {int, str}
```

### 4. Extensible Architecture

Adding support for new type systems is simple:

```python
class MyCustomExtractor(TypeExtractor):
    def can_handle(self, annotation, instance):
        return hasattr(instance, '__my_custom_metadata__')
    
    def extract_type_params(self, annotation):
        return annotation.__my_type_params__
    
    def extract_concrete_types(self, instance):  
        return instance.__my_custom_metadata__['concrete_types']
    
    def get_variance(self, annotation, param_index):
        return Variance.COVARIANT  # or custom logic

# Register with engine
engine.extractors.append(MyCustomExtractor())
```

## Comparison: Original vs Unified

| Feature | Original System | Unified System |
|---------|----------------|----------------|
| **Mixed types** | Immediate failure | Union formation |
| **Variance** | Not supported | Full variance support |
| **TypeVar bounds** | Not enforced | Properly validated |
| **Architecture** | Monolithic | Pluggable extractors |
| **Extensibility** | Hard to extend | Easy to add new types |
| **Error handling** | Binary pass/fail | Graceful degradation |

## Test Results

Running comparison tests shows the improvements:

```bash
$ python test_comparison.py

Testing nested mixed containers: [[1, 2], ['a', 'b']]
✗ Original system failed: Conflicting types for ~A: <class 'int'> vs <class 'str'>
✓ Unified system: str | int

Testing Dict[A, A] with mixed types: {1: 'a', 'b': 2}  
✓ Original system: str | int  # This case happened to work
✓ Unified system: str | int

Testing complex nested branches:
✗ Original system failed: Conflicting types for ~A: <class 'int'> vs <class 'str'>
✓ Unified system: str | int
```

## Future Extensions

The unification framework makes several advanced features easier to implement:

### 1. Callable Type Inference
```python
def apply_func(items: List[A], func: Callable[[A], B]) -> List[B]: ...
# Can infer A from items, B from func signature
```

### 2. Advanced Variance
- Contravariant function parameters
- Bivariant types (rare but theoretically possible)

### 3. Higher-Kinded Types
Support for types that take type constructors as parameters.

### 4. Dependent Types  
Types that depend on runtime values (limited support).

### 5. Effect Systems
Track side effects through type inference.

## Implementation Status

- ✅ **Core unification engine**
- ✅ **Basic constraint collection and solving**  
- ✅ **Variance-aware conflict resolution**
- ✅ **TypeVar bounds/constraint checking**
- ✅ **Pluggable extractor architecture**
- ✅ **Union formation for covariant conflicts**
- ⚠️ **Contravariant supertype inference** (simplified)
- ❌ **Full callable type inference** (future work)
- ❌ **Advanced variance scenarios** (future work)

## Migration Path

1. **Parallel deployment**: Run both systems side-by-side
2. **Gradual migration**: Replace `infer_return_type` with `infer_return_type_unified`
3. **Backward compatibility**: Maintain existing API
4. **Performance testing**: Ensure unification doesn't add significant overhead
5. **Extended testing**: Validate against existing test suite

## Conclusion

The unification-based approach provides:

1. **More robust handling** of real-world type scenarios
2. **Cleaner architecture** that's easier to extend
3. **Proper theoretical foundation** for advanced features
4. **Better error handling** with graceful degradation
5. **Support for advanced type system features**

This makes the type inference system more practical for real applications while providing a foundation for future enhancements. 
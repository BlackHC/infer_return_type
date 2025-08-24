# Infer Return Type Improvements with Generic Utils

## Summary

Updated `infer_return_type.py` to leverage the new and improved `generic_utils` system, resulting in significant improvements in type inference capabilities.

## Key Improvements

### 1. **Simplified Type Extraction**
- Replaced manual type extraction in `_get_concrete_type_for_typevar_binding` with `get_instance_generic_info`
- Now uses `GenericInfo.resolved_type` for consistent type representation
- Eliminates code duplication and manual container type handling

### 2. **Union Type Creation Instead of Conflicts**
- Updated `_bind_typevar_with_conflict_check` to create unions instead of failing on conflicts
- Uses `create_union_if_needed` from generic_utils for consistent union creation
- More robust and useful behavior - instead of failing, creates accurate union types

### 3. **Fixed TypeVar Order Alignment**
- Fixed critical issue where TypeVar binding order was incorrect
- Now uses positional alignment of `concrete_args` instead of relying on `type_params` order
- Correctly handles complex generic dataclasses and Pydantic models

### 4. **Enhanced Nested Structure Handling**
- Improved `_align_nested_structures` to work with GenericInfo objects
- Supports deep nesting like `Wrap[List[Box[A]]]` -> `Wrap[List[Box[int]]]`
- Recursive TypeVar binding through complex nested generic structures

### 5. **Better Substitution Logic**
- Updated `_substitute_type_vars` to use GenericInfo for type reconstruction
- Fallback to manual reconstruction for built-in types when needed
- More reliable type substitution for complex generic types

## Test Results

**Before improvements:** 1 passed, many failed due to basic TypeVar binding issues  
**After improvements:** 39 passed, 0 failed ✨

### Test Status
- ✅ **39 tests passed** (100% success rate) - Complete functionality coverage!
- ✅ **Updated `test_single_typevar_errors`** to reflect improved union creation behavior

### Fixed Issues
- ✅ **Union Type Partitioning**: Fixed `test_mixed_container_multi_typevar` with smart Union handling
- ✅ **Complex Generic Binding**: All dataclass and Pydantic model tests now pass
- ✅ **Nested Structure Inference**: Deep generic nesting works correctly
- ✅ **Test Coverage**: Updated tests to reflect improved behavior (union creation vs failures)

## Technical Changes

### Core Functions Updated
1. `_get_concrete_type_for_typevar_binding()` - Simplified with generic_utils
2. `_bind_typevar_with_conflict_check()` - Union creation instead of errors
3. `_extract_from_custom_generic_unified()` - Fixed alignment logic  
4. `_align_nested_structures()` - Enhanced for GenericInfo objects
5. `_substitute_type_vars()` - Better type reconstruction
6. `_handle_union_container_elements()` - **NEW** Smart Union type partitioning for containers
7. `_type_matches_bound_type()` - **NEW** Helper for Union type matching

### Import Cleanup
- Removed unused imports (`get_origin`, `get_args`, `fields`, etc.)
- Added `create_union_if_needed` from generic_utils
- Streamlined to only needed utilities

## Examples of Improved Behavior

### 1. Mixed Type Containers
```python
def process_mixed(items: List[A]) -> A: pass
result = infer_return_type(process_mixed, [1, "hello", 3.14])
# Before: TypeInferenceError 
# After: int | str | float (union type)
```

### 2. Complex Generic Dataclasses
```python
@dataclass
class Container(Generic[A, B, C]):
    primary: List[A]
    secondary: Dict[str, B] 
    tertiary: Set[C]

def get_primary(c: Container[A, B, C]) -> List[A]: pass
container = Container[int, str, float](...)
result = infer_return_type(get_primary, container)
# Before: Incorrect type binding
# After: list[int] (correct)
```

### 3. Nested Generic Structures
```python
def unwrap_nested(w: Wrap[List[Box[A]]]) -> List[A]: pass
wrapped = Wrap[List[Box[int]]](...)
result = infer_return_type(unwrap_nested, wrapped)
# Before: TypeInferenceError (unbound TypeVar)
# After: list[int] (correct deep inference)
```

### 4. Union Type Partitioning (NEW)
```python
def reorganize_complex(data: List[Tuple[Dict[A, B], Set[A | B]]]) -> Dict[A, List[B]]: pass
complex_data = [
    ({1: "a", 2: "b"}, {1, 2, "a", "b"}),  # Mixed int/str set
    ({3: "c", 4: "d"}, {3, 4, "c", "d"})
]
result = infer_return_type(reorganize_complex, complex_data)
# Before: dict[int, list[int | str]] (incorrect - conflated A and B)
# After: dict[int, list[str]] (correct - A=int from keys, B=str from values)
```

## Benefits

1. **More Robust**: Creates unions instead of failing on conflicts
2. **More Accurate**: Better type inference for complex structures
3. **Simpler Code**: Leverages generic_utils for consistency
4. **Better Coverage**: Handles edge cases that previously failed
5. **Field-based Inference**: Supports inferring types from dataclass/Pydantic field values

The improvements make `infer_return_type` significantly more capable and robust while reducing code complexity through better abstraction with generic_utils.

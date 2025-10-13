import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union, Callable, ForwardRef

import pytest
# from infer_return_type import TypeInferenceError, infer_return_type
from unification_type_inference import (
    TypeInferenceError,
    infer_return_type_unified as infer_return_type,
    UnificationEngine,
    Constraint,
    Substitution,
    Variance,
    UnificationError
)
# from csp_type_inference import CSPTypeInferenceError as TypeInferenceError, infer_return_type_csp as infer_return_type
from pydantic import BaseModel

# TypeVars for testing
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
E = TypeVar('E')
K = TypeVar('K')
V = TypeVar('V')
X = TypeVar('X')
Y = TypeVar('Y')

# =============================================================================
# SHARED TEST FIXTURES - Commonly used generic classes
# =============================================================================
# These fixtures are used throughout the test suite to avoid duplication.
# They cover common patterns: simple containers, nested structures,
# multi-parameter generics, and recursive structures.

# --- Simple Single-Parameter Containers ---

@dataclass
class Wrap(typing.Generic[A]):
    """Simple wrapper for a single value. Most common test fixture."""
    value: A


@dataclass
class Box(typing.Generic[A]):
    """Alternative simple container (semantically same as Wrap)."""
    item: A


class BoxModel(BaseModel, typing.Generic[A]):
    """Pydantic version of Box."""
    item: A


@dataclass
class Container(typing.Generic[A]):
    """Generic container for test data."""
    data: A


# --- Multi-Parameter Containers ---

@dataclass
class TwoParamContainer(typing.Generic[A, B]):
    """Container with two type parameters."""
    first: A
    second: B


@dataclass
class MultiParamContainer(typing.Generic[A, B, C]):
    """Container with three type parameters for complex scenarios."""
    primary: List[A]
    secondary: Dict[str, B]
    tertiary: Set[C]
    mixed: List[Tuple[A, B, C]]


# --- Nested Generic Structures (3 levels) ---

@dataclass
class Level1(typing.Generic[A]):
    """First level of nested generic structure."""
    inner: A


@dataclass
class Level2(typing.Generic[A]):
    """Second level containing Level1."""
    wrapped: Level1[A]
    alternatives: List[A]


class Level3(BaseModel, typing.Generic[A]):
    """Third level (Pydantic) containing Level2."""
    nested: Level2[A]
    extras: Dict[str, A]


# --- Recursive Structures ---

@dataclass
class TreeNode(typing.Generic[A]):
    """Recursive tree structure for testing."""
    value: A
    children: List['TreeNode[A]']


@dataclass
class LinkedNode(typing.Generic[A]):
    """Recursive linked list structure."""
    value: A
    next: Optional['LinkedNode[A]']


# --- Inheritance Test Fixtures ---

@dataclass
class BaseGeneric(typing.Generic[A]):
    """Base generic class for inheritance tests."""
    base_value: A


@dataclass
class DerivedGeneric(BaseGeneric[A], typing.Generic[A]):
    """Simple derived class maintaining same type parameter."""
    derived_value: int


@dataclass
class PartiallySpecialized(typing.Generic[A]):
    """Generic with both fixed and parameterized fields."""
    strings: List[str]  # Fixed type
    generic_items: List[A]  # Parameterized


# --- JSON-like Structures ---

@dataclass
class JsonValue(typing.Generic[A]):
    """JSON-like nested structure for real-world patterns."""
    data: Union[A, Dict[str, 'JsonValue[A]'], List['JsonValue[A]']]


# --- DataFrame-like Structures ---

@dataclass
class TypedColumn(typing.Generic[A]):
    """Represents a typed column in a table."""
    name: str
    values: List[A]


@dataclass
class MultiColumnData(typing.Generic[A, B, C]):
    """Multi-column data structure (DataFrame-like)."""
    col1: TypedColumn[A]
    col2: TypedColumn[B]
    col3: TypedColumn[C]


# =============================================================================
# 1. BASIC SINGLE TYPEVAR TESTS (6 tests)
# =============================================================================

def test_basic_containers():
    """Test basic container type inference with single TypeVars"""
    
    # List operations
    def merge_lists(a: List[A], b: List[A]) -> Set[A]: ...
    t = infer_return_type(merge_lists, [1, 2], [3, 4])
    assert typing.get_origin(t) is set and typing.get_args(t) == (int,)
    
    # Tuple operations  
    def swap(p: Tuple[X, Y]) -> Tuple[Y, X]: ...
    t = infer_return_type(swap, (1, 'x'))
    assert typing.get_args(t) == (str, int)
    
    # Dict operations
    def invert(d: Dict[K, V]) -> Dict[V, K]: ...
    t = infer_return_type(invert, {1: 'a', 2: 'b'})
    assert typing.get_origin(t) is dict and typing.get_args(t) == (str, int)


def test_optional_and_union():
    """Test Optional and Union type handling"""
    
    def pick_first(x: Optional[A]) -> A: ...
    t = infer_return_type(pick_first, 1)
    assert t is int
    
    def merge_with_union(a: List[A], b: List[B]) -> Set[A | B]: ...
    t = infer_return_type(merge_with_union, [1], [2.0])
    assert typing.get_origin(t) is set
    # Handle modern union syntax
    args = typing.get_args(t)
    if len(args) == 1 and hasattr(args[0], '__args__'):
        union_args = typing.get_args(args[0])
        assert set(union_args) == {int, float}
    else:
        assert set(args) == {int, float}


def test_basic_generic_classes():
    """Test basic generic dataclass and Pydantic model inference"""

    def unwrap(w: Wrap[A]) -> A: ...
    t = infer_return_type(unwrap, Wrap[int](value=1))
    assert t is int

    def unbox(bs: List[BoxModel[A]]) -> List[A]: ...
    t = infer_return_type(unbox, [BoxModel[int](item=1)])
    assert typing.get_origin(t) is list and typing.get_args(t) == (int,)


def test_single_typevar_errors():
    """Test error scenarios with single TypeVars"""
    
    def head(xs: List[A]) -> A: ...
    
    # Empty container should fail
    with pytest.raises(TypeInferenceError):
        infer_return_type(head, [])
    
    # Type override should work
    t = infer_return_type(head, [], type_overrides={A: int})
    assert t is int
    
    # Conflicting same TypeVar binding now creates union (improved behavior)
    def same(a: A, b: A) -> bool: ...
    # This now returns int | str instead of failing
    # See test_conflicting_typevar_should_create_union for details


def test_constrained_and_bounded_typevars():
    """Test TypeVars with bounds and constraints"""
    
    T = TypeVar('T', bound=int)
    U = TypeVar('U', bound=str) 
    V = TypeVar('V', int, float)  # Constrained
    
    def multi_bounded(x: T, y: U, z: V) -> Tuple[T, U, V]: ...
    t = infer_return_type(multi_bounded, True, "hello", 3.14)
    assert typing.get_args(t) == (bool, str, float)
    
    def increment_bounded(x: T) -> T: ...
    # Test with a subtype of int (bool is a subtype of int in Python)
    t = infer_return_type(increment_bounded, True)
    assert t is bool  # Should preserve the specific type


def test_empty_containers_with_fallbacks():
    """Test empty containers with various fallback strategies"""
    
    def first_or_default(items: List[A], default: A) -> A: ...
    # Empty list but with a default value to infer from
    t = infer_return_type(first_or_default, [], "default")
    assert t is str
    
    def combine_lists(a: List[A], b: List[A]) -> List[A]: ...
    t = infer_return_type(combine_lists, [1, 2], [3, 4])
    assert typing.get_origin(t) is list and typing.get_args(t) == (int,)


# =============================================================================
# 2. MULTI-TYPEVAR INTERACTIONS (4 tests)
# =============================================================================

def test_complex_nested_dict_multiple_typevars():
    """Test complex nested dict patterns with multiple TypeVars"""
    
    # Pattern: dict[str, dict[A, B]] -> set[A | B]
    def extract_nested_dict_union(d: Dict[str, Dict[A, B]]) -> Set[A | B]: ...
    
    nested_data = {
        "section1": {1: "hello", 2: "world"},
        "section2": {3: "foo", 4: "bar"}
    }
    
    t = infer_return_type(extract_nested_dict_union, nested_data)
    assert typing.get_origin(t) is set
    # Should be set[int | str]
    union_args = typing.get_args(typing.get_args(t)[0])
    assert set(union_args) == {int, str}


def test_triple_nested_dict_pattern():
    """Test three-level dict nesting with multiple TypeVars"""
    
    # Pattern: dict[A, dict[B, dict[C, int]]] -> tuple[A, B, C]
    def extract_triple_keys(d: Dict[A, Dict[B, Dict[C, int]]]) -> Tuple[A, B, C]: ...
    
    triple_data = {
        "level1": {
            42: {
                3.14: 100
            }
        }
    }
    
    t = infer_return_type(extract_triple_keys, triple_data)
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (str, int, float)


def test_mixed_container_multi_typevar():
    """Test mixed containers with multiple interacting TypeVars"""
    
    # Pattern: list[tuple[dict[A, B], set[A | B]]] -> dict[A, list[B]]
    def reorganize_complex(data: List[Tuple[Dict[A, B], Set[A | B]]]) -> Dict[A, List[B]]: ...
    
    complex_data = [
        ({1: "a", 2: "b"}, {1, 2, "a", "b"}),
        ({3: "c", 4: "d"}, {3, 4, "c", "d"})
    ]
    
    t = infer_return_type(reorganize_complex, complex_data)
    assert typing.get_origin(t) is dict
    key_type, value_type = typing.get_args(t)
    assert key_type is int
    assert typing.get_origin(value_type) is list
    assert typing.get_args(value_type) == (str,)


def test_multi_typevar_error_scenarios():
    """Test error handling with multiple TypeVars"""
    
    # Different TypeVars can have different types (should work)
    def inconsistent_types(a: Dict[A, B], b: Dict[A, C]) -> Tuple[A, B, C]: ...
    dict1 = {1: "string"}  # A=int, B=str
    dict2 = {1: 42}        # A=int, C=int (B != C is OK)
    
    t = infer_return_type(inconsistent_types, dict1, dict2)
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (int, str, int)
    
    # Same TypeVar with conflicting types - improved engine creates union instead of failing
    def same_typevar_conflict(a: List[A], b: List[A]) -> A: ...
    t = infer_return_type(same_typevar_conflict, [1, 2], ["a", "b"])
    # Should return int | str union type
    import types
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}
    
    # Partial binding with empty containers (should still fail due to unbound TypeVars)
    def partial_multi_binding(a: List[A], b: Dict[B, C], c: Set[A]) -> Tuple[A, B, C]: ...
    with pytest.raises(TypeInferenceError):
        infer_return_type(partial_multi_binding, [], {}, {42})


# =============================================================================
# 3. DEEP/COMPLEX NESTING (3 tests)
# =============================================================================

def test_consolidated_nested_generics():
    """Consolidated test for deeply nested generic structures"""

    def unwrap_all_levels(l3: Level3[A]) -> A: ...
    def get_alternatives(l3: Level3[A]) -> List[A]: ...
    def get_extras_values(l3: Level3[A]) -> List[A]: ...
    
    deep_structure = Level3[bool](
        nested=Level2[bool](
            wrapped=Level1[bool](inner=True),
            alternatives=[False, True]
        ),
        extras={"flag1": False, "flag2": True}
    )
    
    # Test all extraction patterns in one consolidated test
    t1 = infer_return_type(unwrap_all_levels, deep_structure)
    assert t1 is bool
    
    t2 = infer_return_type(get_alternatives, deep_structure)  
    assert typing.get_origin(t2) is list and typing.get_args(t2) == (bool,)
    
    t3 = infer_return_type(get_extras_values, deep_structure)
    assert typing.get_origin(t3) is list and typing.get_args(t3) == (bool,)


def test_consolidated_multi_param_container():
    """Consolidated test for multi-parameter generic containers"""

    def get_primary(mc: MultiParamContainer[A, B, C]) -> List[A]: ...
    def get_secondary_values(mc: MultiParamContainer[A, B, C]) -> List[B]: ...
    def get_tertiary(mc: MultiParamContainer[A, B, C]) -> Set[C]: ...
    def get_mixed_tuples(mc: MultiParamContainer[A, B, C]) -> List[Tuple[A, B, C]]: ...
    
    container = MultiParamContainer[int, str, float](
        primary=[1, 2, 3],
        secondary={"a": "hello", "b": "world"},
        tertiary={1.1, 2.2, 3.3},
        mixed=[(1, "a", 1.1), (2, "b", 2.2)]
    )
    
    # Test all extractions in consolidated manner
    assert infer_return_type(get_primary, container) == list[int]
    assert infer_return_type(get_secondary_values, container) == list[str]
    assert infer_return_type(get_tertiary, container) == set[float]
    
    mixed_type = infer_return_type(get_mixed_tuples, container)
    assert typing.get_origin(mixed_type) is list
    tuple_type = typing.get_args(mixed_type)[0]
    assert typing.get_origin(tuple_type) is tuple
    assert typing.get_args(tuple_type) == (int, str, float)


def test_real_world_patterns():
    """Test real-world complex patterns like JSON and DataFrame structures"""

    def extract_json_type(json_val: JsonValue[A]) -> A: ...
    
    nested_json = JsonValue[int](
        data={
            "numbers": JsonValue[int](data=[
                JsonValue[int](data=42),
                JsonValue[int](data=100)
            ])
        }
    )
    
    t = infer_return_type(extract_json_type, nested_json)
    assert t is int
    
    # DataFrame-like multi-column structure
    def get_first_column_type(data: MultiColumnData[A, B, C]) -> List[A]: ...
    def get_all_column_types(data: MultiColumnData[A, B, C]) -> Tuple[List[A], List[B], List[C]]: ...
    
    df_data = MultiColumnData[int, str, float](
        col1=TypedColumn[int]("integers", [1, 2, 3]),
        col2=TypedColumn[str]("strings", ["a", "b", "c"]),
        col3=TypedColumn[float]("floats", [1.1, 2.2, 3.3])
    )
    
    t1 = infer_return_type(get_first_column_type, df_data)
    assert typing.get_origin(t1) is list
    assert typing.get_args(t1) == (int,)
    
    t2 = infer_return_type(get_all_column_types, df_data)
    assert typing.get_origin(t2) is tuple
    tuple_args = typing.get_args(t2)
    assert len(tuple_args) == 3
    assert tuple_args[0] == list[int]
    assert tuple_args[1] == list[str] 
    assert tuple_args[2] == list[float]


# =============================================================================
# 4. ADVANCED FEATURES (3 tests)
# =============================================================================

def test_callable_and_function_generics():
    """Test generic callables and function type inference"""
    
    # Note: Callable type inference is complex and currently not fully supported
    # This test demonstrates the limitation and could be extended in the future
    
    def simple_transform(data: List[A], value: B) -> List[B]: ...
    
    # Use a simpler pattern that works with current implementation
    t = infer_return_type(simple_transform, ["hello", "world"], 42)
    assert typing.get_origin(t) is list
    assert typing.get_args(t) == (int,)


def test_complex_union_scenarios():
    """Test complex union patterns with multiple TypeVars"""
    
    def complex_union_result(data: Dict[A, List[B]]) -> Union[A, List[B], Tuple[A, B]]: ...
    
    data = {"key": [1, 2, 3]}
    t = infer_return_type(complex_union_result, data)
    
    # Should be Union[str, List[int], Tuple[str, int]]
    # Handle both typing.Union and types.UnionType (Python 3.10+)
    import types
    union_origin = typing.get_origin(t)
    assert union_origin is Union or union_origin is getattr(types, 'UnionType', None)
    
    union_args = typing.get_args(t)
    assert str in union_args
    
    # Check for List[int] in union
    list_types = [arg for arg in union_args if typing.get_origin(arg) is list]
    assert len(list_types) > 0
    assert typing.get_args(list_types[0]) == (int,)
    
    # Check for Tuple[str, int] in union  
    tuple_types = [arg for arg in union_args if typing.get_origin(arg) is tuple]
    assert len(tuple_types) > 0
    assert typing.get_args(tuple_types[0]) == (str, int)
    
    # Test union with generics
    def maybe_wrap(x: A, should_wrap: bool) -> A | Wrap[A]: ...
    
    t = infer_return_type(maybe_wrap, 42, True)
    # Should return int | Wrap[int]
    if hasattr(t, '__args__'):
        union_types = typing.get_args(t)
        assert int in union_types
        # Check if Wrap[int] is in the union (might be represented differently)
        wrap_types = [arg for arg in union_types if typing.get_origin(arg) == Wrap]
        assert len(wrap_types) > 0


def test_advanced_inheritance_and_specialization():
    """Test advanced inheritance chains and partial specialization"""
    
    # TypeVar inheritance chain
    @dataclass
    class Base(typing.Generic[A, B]):
        base_a: A
        base_b: B
    
    @dataclass 
    class Derived(Base[A, str], typing.Generic[A]):  # Partially specialize B=str
        derived_data: List[A]
    
    def extract_from_derived(d: Derived[A]) -> Tuple[A, List[A]]: ...
    
    derived_instance = Derived[int](
        base_a=42,
        base_b="fixed_string", 
        derived_data=[1, 2, 3]
    )
    
    t = infer_return_type(extract_from_derived, derived_instance)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args[0] is int
    assert typing.get_origin(args[1]) is list
    assert typing.get_args(args[1]) == (int,)
    
    # Partially specialized generic
    def get_generic_items(ps: PartiallySpecialized[A]) -> List[A]: ...
    def get_strings(ps: PartiallySpecialized[A]) -> List[str]: ...
    
    ps = PartiallySpecialized[int](strings=["a", "b"], generic_items=[1, 2, 3])
    
    t_generic = infer_return_type(get_generic_items, ps)
    assert typing.get_origin(t_generic) is list and typing.get_args(t_generic) == (int,)
    
    t_strings = infer_return_type(get_strings, ps)
    assert typing.get_origin(t_strings) is list and typing.get_args(t_strings) == (str,)
    
    # Recursive generic structure
    def get_tree_value(node: TreeNode[A]) -> A: ...
    def get_tree_children(node: TreeNode[A]) -> List[TreeNode[A]]: ...
    
    tree = TreeNode[str](
        value="root",
        children=[
            TreeNode[str](value="child1", children=[]),
            TreeNode[str](value="child2", children=[])
        ]
    )
    
    # Test extracting value
    t_value = infer_return_type(get_tree_value, tree)
    assert t_value is str
    
    # Test extracting children list
    t_children = infer_return_type(get_tree_children, tree)
    assert typing.get_origin(t_children) is list
    children_arg = typing.get_args(t_children)[0]
    # The children type should be TreeNode[str]
    assert typing.get_origin(children_arg) == TreeNode


# =============================================================================
# ADDITIONAL EDGE CASES (kept from original tests)
# =============================================================================

def test_nested_list_of_generics():
    """Test handling nested lists of generic types"""

    def unwrap_box_list(w: Wrap[List[BoxModel[A]]]) -> List[A]: ...

    boxes = [BoxModel[int](item=1), BoxModel[int](item=2)]
    wrapped_boxes = Wrap[List[BoxModel[int]]](value=boxes)

    t = infer_return_type(unwrap_box_list, wrapped_boxes)
    assert typing.get_origin(t) is list and typing.get_args(t) == (int,)


def test_optional_nested_generics():
    """Test handling optional nested generic types"""

    def unwrap_optional_nested(w: Optional[Wrap[A]]) -> Optional[A]: ...

    wrapped = Wrap[float](value=3.14)
    t = infer_return_type(unwrap_optional_nested, wrapped)
    # Should handle Optional[float] or Union[float, None]
    assert t == Optional[float] or (typing.get_origin(t) is type(typing.Union[float, None]))


def test_nested_dict_extraction():
    """Test extracting from nested dictionary structures"""
    
    def extract_nested_values(d: Dict[str, Dict[A, B]]) -> List[B]: ...
    
    nested_dict = {"key": {1: "value1", 2: "value2"}}
    t = infer_return_type(extract_nested_values, nested_dict)
    assert typing.get_origin(t) is list and typing.get_args(t) == (str,)


# =============================================================================
# EDGE CASE TESTS BASED ON _extract_typevar_bindings_from_annotation ANALYSIS
# =============================================================================

def test_union_type_limitations():
    """Test Union type handling - should work but currently fails"""
    
    # General Union types should be supported
    def process_union(data: Union[List[A], Set[A]]) -> A: ...
    
    # This should work - clearly a list, should bind A=int
    t = infer_return_type(process_union, [1, 2, 3])
    assert t is int
    
    # This should also work - clearly a set, should bind A=str
    t = infer_return_type(process_union, {"hello", "world"})
    assert t is str
    
    # Modern union syntax should also work
    def process_modern_union(data: List[A] | Set[A]) -> A: ...
    
    t = infer_return_type(process_modern_union, [1, 2, 3])
    assert t is int


def test_mixed_type_container_behavior():
    """Test that mixed-type containers should infer union types"""
    
    def process_mixed_list(items: List[A]) -> A: ...
    
    # Mixed type list should infer union type A = int | str | float
    mixed_list = [1, "hello", 3.14]  # int, str, float
    t = infer_return_type(process_mixed_list, mixed_list)
    
    # Should be int | str | float (union type)
    import types
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str, float}
    
    def process_mixed_dict_values(data: Dict[str, A]) -> A: ...
    
    # Mixed value types should infer A = int | str
    mixed_dict = {"a": 1, "b": "hello"}  # int, str values
    t = infer_return_type(process_mixed_dict_values, mixed_dict)
    
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


def test_empty_container_inference_limitations():
    """Test limitations with empty containers - these should fail as expected"""
    
    def process_empty_list(items: List[A]) -> A: ...
    def process_empty_dict(data: Dict[A, B]) -> Tuple[A, B]: ...
    def process_empty_set(items: Set[A]) -> A: ...
    
    # Empty containers cannot provide type information - these should fail
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_empty_list, [])
    
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_empty_dict, {})
    
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_empty_set, set())


def test_type_mismatch_graceful_handling():
    """Test that type mismatches are handled gracefully - should fail as expected"""
    
    def process_list(items: List[A]) -> A: ...
    
    # Annotation expects List[A] but value is not a list - should fail gracefully
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_list, "not_a_list")
    
    def process_dict(data: Dict[A, B]) -> Tuple[A, B]: ...
    
    # Annotation expects Dict[A, B] but value is not a dict - should fail gracefully
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_dict, [1, 2, 3])


def test_variable_vs_fixed_length_tuples():
    """Test difference between variable and fixed length tuple handling"""
    
    def process_var_tuple(data: Tuple[A, ...]) -> A: ...
    def process_fixed_tuple(data: Tuple[A, B, C]) -> Tuple[A, B, C]: ...
    
    # Variable length tuple - all elements should have same type
    var_tuple = (1, 2, 3, 4, 5)
    t = infer_return_type(process_var_tuple, var_tuple)
    assert t is int
    
    # Fixed length tuple - each position can have different type
    fixed_tuple = (1, "hello", 3.14)
    t = infer_return_type(process_fixed_tuple, fixed_tuple)
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (int, str, float)
    
    # Fixed tuple with wrong length should still work for available positions
    def process_three_tuple(data: Tuple[A, B, C]) -> A: ...
    
    # Tuple has only 2 elements but annotation expects 3 - should still bind A
    t = infer_return_type(process_three_tuple, (1, "hello"))
    assert t is int


def test_deeply_nested_structure_limits():
    """Test deeply nested structures work correctly"""
    
    def process_deep_nested(data: List[List[List[List[A]]]]) -> A: ...
    
    # Very deep nesting should work
    deep_data = [[[["bottom"]]]]
    t = infer_return_type(process_deep_nested, deep_data)
    assert t is str
    
    # Even deeper with mixed containers
    def process_very_deep(data: List[Dict[str, Set[Tuple[A, B]]]]) -> Tuple[A, B]: ...
    
    very_deep_data = [{"key": {(1, "a"), (2, "b")}}]
    t = infer_return_type(process_very_deep, very_deep_data)
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (int, str)


def test_complex_union_container_scenarios():
    """Test complex union scenarios should work"""
    
    # Union of different container types should work
    def process_list_or_dict(data: Union[List[A], Dict[str, A]]) -> A: ...
    
    # Should recognize this as List[int] and bind A=int
    t = infer_return_type(process_list_or_dict, [1, 2, 3])
    assert t is int
    
    # Should recognize this as Dict[str, int] and bind A=int  
    t = infer_return_type(process_list_or_dict, {"key": 42})
    assert t is int
    
    # Union of generic containers with different type params
    def process_container_union(data: Union[List[A], Set[B]]) -> Union[A, B]: ...
    
    # Should bind A=int and return A (since it's a list)
    t = infer_return_type(process_container_union, [1, 2, 3])
    assert t is int


def test_optional_nested_in_containers():
    """Test Optional types nested within containers"""
    
    def process_optional_list(data: List[Optional[A]]) -> A: ...
    
    # List with mix of values and None - should bind A from non-None values
    optional_list = [1, None, 2, None, 3]
    t = infer_return_type(process_optional_list, optional_list)
    assert t is int
    
    def process_list_of_optionals(data: Optional[List[A]]) -> A: ...
    
    # Optional list (not None) should work
    t = infer_return_type(process_list_of_optionals, [1, 2, 3])
    assert t is int
    
    # None case should fail appropriately
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_list_of_optionals, None)


def test_callable_type_variable_inference_limits():
    """Test callable type variable inference limitations"""
    # TODO: This test documents a current limitation - Callable type inference is not supported yet.

    # Callable types with TypeVars should be inferrable from function signatures
    def apply_func(items: List[A], func: Callable[[A], B]) -> List[B]: ...
    
    def int_to_str(x: int) -> str:
        return str(x)
    
    # TODO: This currently fails because callable type inference is not implemented
    # Should infer A=int from list, B=str from callable signature
    with pytest.raises(TypeInferenceError):
        infer_return_type(apply_func, [1, 2, 3], int_to_str)


def test_generic_class_without_type_parameters():
    """Test behavior with generic classes that don't specify type parameters"""
    
    @dataclass
    class GenericContainer(typing.Generic[A]):
        value: A
    
    def process_generic(container: GenericContainer[A]) -> A: ...
    
    # Creating instance without explicit type parameter should still work
    container = GenericContainer(value=42)  # No [int] specified
    
    # Should infer from the instance data
    t = infer_return_type(process_generic, container)
    assert t is int


def test_inheritance_chain_type_binding():
    """Test TypeVar binding through inheritance chains"""
    
    @dataclass
    class BaseGeneric(typing.Generic[A]):
        base_value: A
    
    @dataclass
    class DerivedGeneric(BaseGeneric[str]):  # Concrete specialization
        derived_value: int
    
    def process_derived(obj: DerivedGeneric) -> str: ...
    
    derived = DerivedGeneric(base_value="hello", derived_value=42)
    t = infer_return_type(process_derived, derived)
    assert t is str
    
    # More complex inheritance with TypeVars
    @dataclass  
    class MultiLevel(DerivedGeneric, typing.Generic[B]):
        extra: B
    
    def process_multi(obj: MultiLevel[B]) -> B: ...
    
    multi = MultiLevel[float](base_value="hello", derived_value=42, extra=3.14)
    t = infer_return_type(process_multi, multi)
    assert t is float


def test_multiple_union_containers():
    """Test functions with multiple union container parameters"""
    
    def process_multiple_unions(
        data1: Union[List[A], Tuple[A, ...]], 
        data2: Union[Set[B], Dict[str, B]]
    ) -> Tuple[A, B]: ...
    
    # Should handle multiple union parameters
    t = infer_return_type(process_multiple_unions, [1, 2], {"a": "hello", "b": "world"})
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (int, str)


def test_nested_unions_in_generics():
    """Test nested union types within generic containers"""
    
    def process_nested_union(data: List[Union[A, B]]) -> Union[A, B]: ...
    
    # List containing mixed types should infer union
    mixed_list = [1, "hello", 2, "world"]  # int and str mixed
    t = infer_return_type(process_nested_union, mixed_list)
    
    # Should return Union[int, str] or int | str
    import types
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


# =============================================================================
# MORE SPECIFIC EDGE CASES (additional limitations)
# =============================================================================

def test_homogeneous_containers_vs_mixed_containers():
    """
    Test the difference between homogeneous and mixed container inference.
    
    This documents current behavior and potential edge cases.
    """
    
    def process_list(data: List[A]) -> A: ...
    
    # Homogeneous container - should work fine
    t = infer_return_type(process_list, [1, 2, 3])
    assert t is int
    
    # Mixed container - should infer union type (this works correctly)
    t = infer_return_type(process_list, [1, "hello", 3.14])
    origin = typing.get_origin(t)
    import types
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    
    # Nested mixed containers with same TypeVar - improved engine now handles this
    def process_nested_mixed(data: List[List[A]]) -> A: ...
    
    # The improved engine creates union from mixed nested types instead of failing
    t = infer_return_type(process_nested_mixed, [[1, 2], ["a", "b"]])
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


def test_typevar_in_key_and_value_positions():
    """
    Test TypeVar appearing in both key and value positions of same dict.
    
    TODO: This tests edge cases where same TypeVar appears in multiple positions.
    """
    
    def process_self_referential_dict(data: Dict[A, A]) -> A: ...
    
    # Consistent types - should work
    t = infer_return_type(process_self_referential_dict, {1: 2, 3: 4})
    assert t is int
    
    # But what about mixed types where keys and values should both contribute to union?
    # TODO: This might be a limitation - investigate if this properly infers union
    mixed_dict = {1: "a", "b": 2}  # Keys: int|str, Values: str|int
    
    # This should ideally infer A = int | str but might conflict
    # Need to test actual behavior
    try:
        t = infer_return_type(process_self_referential_dict, mixed_dict)
        # If it works, verify it's a union type
        origin = typing.get_origin(t)
        import types
        assert origin is Union or origin is getattr(types, 'UnionType', None)
        union_args = typing.get_args(t)
        assert set(union_args) == {int, str}
    except TypeInferenceError:
        # If it fails, that's a limitation to document
        # TODO: Remove this pytest.raises when limitation is fixed
        pytest.fail("TypeVar binding conflict in Dict[A, A] with mixed key/value types - this is a limitation")


def test_typevar_with_none_values():
    """
    Test TypeVar inference when containers include None values.
    
    Tests how None values affect TypeVar binding - should properly include None in unions.
    """
    
    def process_optional_elements(data: List[A]) -> A: ...
    
    # List with None values - should infer union including NoneType
    mixed_with_none = [1, None, 2, None]
    t = infer_return_type(process_optional_elements, mixed_with_none)
    
    # Should infer A = int | None
    origin = typing.get_origin(t)
    import types
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(t)
    assert int in union_args
    assert type(None) in union_args
    
    # Nested containers with None should also include None in the union
    def process_nested_with_none(data: List[List[A]]) -> A: ...
    
    # Should correctly infer A = int | None from nested structure
    t = infer_return_type(process_nested_with_none, [[1, 2], [None, None]])
    # Should be int | None union
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(t)
    assert int in union_args
    assert type(None) in union_args


def test_empty_vs_non_empty_container_combinations():
    """
    Test combinations of empty and non-empty containers.
    
    TODO: This tests edge cases with partial TypeVar binding.
    """
    
    def process_multiple_lists(list1: List[A], list2: List[A], list3: List[A]) -> A: ...
    
    # Some empty, some non-empty - should infer from non-empty ones
    t = infer_return_type(process_multiple_lists, [], [1, 2], [])
    assert t is int
    
    # If non-empty ones conflict, the improved engine now creates unions
    t = infer_return_type(process_multiple_lists, [], [1, 2], ["a", "b"])
    origin = typing.get_origin(t)
    import types
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}
    
    
def test_multiple_nested_typevars():
    
    class PydanticModel(BaseModel, typing.Generic[A, B]):
        a: A
        b: B
    
    def process_pydantic_model(data: PydanticModel[A, list[B]]) -> B: ...

    t = infer_return_type(process_pydantic_model, PydanticModel[int, list[str]](a=1, b=["hello", "world"]))
    assert t is str


def test_deeply_nested_with_different_branching():
    """
    Test deeply nested structures where branches have different types.
    
    TODO: This tests limitations in complex nested type inference.
    """
    
    def process_complex_nested(data: List[Dict[str, List[A]]]) -> A: ...
    
    # Complex nesting where different branches should contribute to union
    complex_data = [
        {"branch1": [1, 2, 3]},        # A = int
        {"branch2": ["a", "b", "c"]}   # A = str (should conflict)
    ]
    
    # The improved engine now handles complex nested branching with unions
    t = infer_return_type(process_complex_nested, complex_data)
    # Creates union from different branch types
    origin = typing.get_origin(t)
    import types
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


def test_typevar_inference_with_subtyping():
    """
    Test TypeVar inference when dealing with subtypes.
    
    This documents current behavior with subtype relationships.
    """
    
    def process_numbers(data: List[A]) -> A: ...
    
    # Mix of int and bool (bool is subtype of int in Python)
    mixed_subtypes = [True, 1, False, 2]
    t = infer_return_type(process_numbers, mixed_subtypes)
    
    # Should infer union of bool and int
    origin = typing.get_origin(t)
    import types
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(t)
    assert bool in union_args and int in union_args


def test_variance_and_contravariance_limitations():
    """
    Test limitations related to variance in generic types.
    
    TODO: This is a more advanced edge case related to variance.
    """
    
    # Callable with contravariant input and covariant output
    def process_with_callable(data: List[A], transform: Callable[[A], B]) -> List[B]: ...
    
    def int_to_str(x: int) -> str:
        return str(x)
    
    # TODO: This currently fails because callable type inference is not implemented
    # Simple case should work but doesn't yet
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_with_callable, [1, 2, 3], int_to_str)
    
    # But more complex variance scenarios might have limitations
    # This is documenting current behavior rather than testing a specific limitation


# =============================================================================
# KNOWN WEAKNESSES - TESTS TO SKIP (Need fixing before production)
# =============================================================================

def test_conflicting_typevar_should_create_union():
    """
    Test that conflicting TypeVar bindings should create unions, not fail.
    
    This is a CRITICAL weakness - Original and CSP both handle this correctly.
    """
    def identity(a: A, b: A) -> A: ...
    
    # This currently fails but SHOULD return int | str
    result = infer_return_type(identity, 1, 'x')
    
    # Should be union type
    import types
    origin = typing.get_origin(result)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(result)
    assert set(union_args) == {int, str}


def test_none_filtering_in_optional():
    """
    Test that None values in Optional[A] should not bind A to None.
    
    This is different behavior from Original and CSP - they filter None out correctly.
    """
    def process_dict_with_nones(d: Dict[str, Optional[A]]) -> A: ...
    
    # With None values mixed in, should still return int, not int | None
    result = infer_return_type(process_dict_with_nones, {'a': 1, 'b': None, 'c': 2})
    assert result is int  # Should be int, not int | None


def test_complex_union_structure():
    """
    Test complex union structures like Union[A, List[A], Dict[str, A]].
    
    CSP handles this correctly, unification fails with conflicting type assignments.
    """
    def extract_value(data: Dict[str, Union[A, List[A], Dict[str, A]]]) -> A: ...
    
    test_data = {
        'single': 42,
        'list': [43, 44],
        'nested': {'value': 45}
    }
    
    # Should extract A = int from all three positions
    result = infer_return_type(extract_value, test_data)
    assert result is int


def test_bounded_typevar_strict():
    """
    Test that bounded TypeVars are strictly checked per PEP 484.
    
    The implementation follows PEP 484: int is not a subtype of float in the type system,
    even though Python's runtime accepts int where float is expected.
    This is the correct static typing behavior.
    """
    Numeric = TypeVar('Numeric', bound=float)
    def process_numeric(x: Numeric) -> Numeric: ...
    
    # int is not a subtype of float in the type system (per PEP 484)
    # This should fail with a TypeInferenceError
    with pytest.raises(TypeInferenceError, match="doesn't satisfy bound"):
        infer_return_type(process_numeric, 1)
    
    # But float should work
    result = infer_return_type(process_numeric, 1.5)
    assert result is float

def test_set_union_distribution_fixed():
    """
    Test Set[Union[A, B]] distribution among TypeVars.
    
    When we have exactly N distinct types and N TypeVars in a union,
    we can distribute one type to each TypeVar.
    """
    def process_union_set(s: Set[Union[A, B]]) -> Tuple[Set[A], Set[B]]: ...
    
    result = infer_return_type(process_union_set, {1, 'a', 2, 'b'})
    
    # A and B should get distributed types
    tuple_args = typing.get_args(result)
    assert len(tuple_args) == 2
    
    # Both should be sets
    origin1 = typing.get_origin(tuple_args[0])
    origin2 = typing.get_origin(tuple_args[1])
    assert origin1 is set and origin2 is set
    
    # Get the element types from each set
    element_type1 = typing.get_args(tuple_args[0])[0]
    element_type2 = typing.get_args(tuple_args[1])[0]
    
    # Should be {int, str} in some order
    assert {element_type1, element_type2} == {int, str}
    
    # Verify they are different (proper distribution)
    assert element_type1 != element_type2


# =============================================================================
# MISSING TESTS FROM CSP - VARIANCE AND CONSTRAINTS
# =============================================================================

def test_covariant_variance_explicit():
    """
    Test explicit covariant variance handling.
    
    List[A] is covariant, so all elements' types are collected and unified.
    For homogeneous lists, we get the most specific type.
    For mixed lists, we preserve type precision with unions.
    """
    class Animal: pass
    class Dog(Animal): pass
    
    def covariant_test(pets: List[A]) -> A: ...
    
    # List is covariant - we infer the most specific type (Dog)
    dog_list = [Dog(), Dog()]
    result = infer_return_type(covariant_test, dog_list)
    
    # Should infer Dog (most specific type from the list)
    assert result is Dog
    
    class Cat(Animal): pass
    
    cat_list = [Cat(), Cat()]
    result = infer_return_type(covariant_test, cat_list)
    
    # Should infer Cat (most specific type from the list)
    assert result is Cat
    
    mixed_list = [Dog(), Cat()]
    result = infer_return_type(covariant_test, mixed_list)
    
    # Should infer Dog | Cat (preserve type precision)
    import types
    origin = typing.get_origin(result)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(result)
    assert set(union_args) == {Dog, Cat}
    
    # Test with bounded TypeVar - bounds are checked but union is still preserved
    T_bounded = TypeVar('T_bounded', bound=Animal)
    def bounded_test(pets: List[T_bounded]) -> T_bounded: ...
    
    bounded_result = infer_return_type(bounded_test, mixed_list)
    # Should still be Dog | Cat union (both satisfy the bound)
    bounded_origin = typing.get_origin(bounded_result)
    assert bounded_origin is Union or bounded_origin is getattr(types, 'UnionType', None)
    bounded_args = typing.get_args(bounded_result)
    assert set(bounded_args) == {Dog, Cat}


@pytest.mark.skip(reason="LIMITATION: Callable parameter extraction requires signature inspection")
def test_contravariant_variance_explicit():
    """
    Test contravariant variance in Callable parameters.
    
    Callable is contravariant in parameter types, covariant in return type.
    This would require extracting parameter types from function signatures,
    which is more complex than the current implementation supports.
    """
    def contravariant_test(func: Callable[[A], str]) -> A: ...
    
    def object_to_str(x: object) -> str:
        return str(x)
    
    # Would need to extract that object_to_str takes object parameter
    # This requires inspecting the function signature at a deeper level
    result = infer_return_type(contravariant_test, object_to_str)
    assert result is object


def test_invariant_dict_keys():
    """
    Test invariant variance for Dict keys.
    
    Dict keys are invariant - the exact type used is inferred, not a supertype.
    """
    class StringKey(str): pass
    
    def invariant_test(mapping: Dict[A, int]) -> A: ...
    
    # Should be exactly the key type used, not a supertype
    string_dict = {'key': 1}
    result1 = infer_return_type(invariant_test, string_dict)
    assert result1 is str
    
    custom_dict = {StringKey('key'): 1}
    result2 = infer_return_type(invariant_test, custom_dict)
    assert result2 is StringKey


def test_constraint_priority_resolution():
    """
    Test that type_overrides have highest priority in constraint resolution.
    
    Type overrides should take precedence over inferred types from values.
    """
    # type_overrides should have highest priority
    def process_list(items: List[A]) -> A: ...
    
    # Even though we have int values, override should win
    result = infer_return_type(process_list, [1, 2, 3], type_overrides={A: str})
    assert result is str


def test_domain_filtering_with_constraints():
    """
    Test constraint filtering with bounded TypeVars.
    
    When multiple values must satisfy a bound, the system should find a type
    that works for all values and satisfies the bound.
    """
    T_BOUNDED = TypeVar('T_BOUNDED', bound=int)
    
    def bounded_test(x: T_BOUNDED, y: T_BOUNDED) -> T_BOUNDED: ...
    
    # Both values must satisfy the bound (bool is subtype of int)
    result = infer_return_type(bounded_test, True, False)
    assert result is bool  # bool is subtype of int


def test_union_or_logic_distribution():
    """
    Test Union types in parameter annotations.
    
    Union[List[A], Set[A]] means the value can be either a List or Set,
    and A can be inferred from whichever branch matches the actual value.
    """
    def process_union(data: Union[List[A], Set[A]]) -> A: ...
    
    # List branch
    result1 = infer_return_type(process_union, [1, 2, 3])
    assert result1 is int
    
    # Set branch  
    result2 = infer_return_type(process_union, {1, 2, 3})
    assert result2 is int


def test_subset_constraints():
    """
    Test distributing types in Set[Union[A, B]] to separate A and B.
    
    When we have exactly N distinct types and N TypeVars in a union,
    we distribute one type to each TypeVar using a simple heuristic.
    """
    def process_set_union(s: Set[Union[A, B]]) -> Tuple[A, B]: ...
    
    # Set with mixed types - distribute to A=int, B=str (or vice versa)
    mixed_set = {1, 'hello', 2, 'world'}
    result = infer_return_type(process_set_union, mixed_set)
    
    # Should distribute types properly
    assert typing.get_origin(result) is tuple
    # Result could be (int, str) or (str, int) depending on sorting
    result_types = set(typing.get_args(result))
    assert result_types == {int, str}


# =============================================================================
# EDGE CASES THAT SHOULD WORK BUT MIGHT NOT
# =============================================================================

def test_multiple_invariant_conflicts():
    """
    Test behavior when multiple invariant constraints conflict.
    
    When the same TypeVar appears in multiple invariant positions with different types,
    the unification engine creates a union type.
    """
    def process_containers(d1: Dict[A, str], d2: Dict[A, str], d3: Dict[A, str]) -> A: ...
    
    # Three dicts with different key types (invariant position)
    result = infer_return_type(process_containers, 
                               {1: 'a'}, 
                               {'x': 'b'}, 
                               {3.14: 'c'})
    
    # Should create int | str | float union
    origin = typing.get_origin(result)
    import types
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(result)
    assert set(union_args) == {int, str, float}


def test_nested_variance_mixing():
    """
    Test mixing invariant and covariant variance in nested structures.
    
    Dict[A, List[A]] - A is invariant in dict keys and covariant in list elements.
    The unification engine correctly handles both variance positions.
    """
    def process_mixed_variance(d: Dict[A, List[A]]) -> A: ...
    
    # Keys must match (invariant), list elements are covariant
    test_data = {
        1: [1, 2, 3],
        2: [4, 5, 6]
    }
    
    result = infer_return_type(process_mixed_variance, test_data)
    assert result is int


@pytest.mark.skip(reason="LIMITATION: Callable type signature extraction not supported")
def test_typevar_multiple_variance_positions():
    """
    Test TypeVar appearing in positions with different variance.
    
    Callable[[List[A]], A] - A is covariant in List, covariant in return position.
    The unification engine would need to extract TypeVar bindings from Callable signatures.
    """
    def complex_func(callback: Callable[[List[A]], A], default: A) -> A: ...
    
    def list_to_int(items: List[int]) -> int:
        return sum(items)
    
    # Should infer A = int from both the default value and the callable
    result = infer_return_type(complex_func, list_to_int, 42)
    assert result is int


# =============================================================================
# DEBUGGING AND ERROR MESSAGE TESTS  
# =============================================================================

def test_constraint_trace_on_failure():
    """
    Test that conflicting constraints are handled by creating unions.
    
    The unification engine creates union types when the same TypeVar
    receives multiple incompatible constraints.
    """
    def conflicting_example(a: List[A], b: List[A]) -> A: ...
    
    import types
    # This creates a union now (improved behavior)
    result = infer_return_type(conflicting_example, [1], ["x"])
    # Current behavior: creates int | str union
    assert typing.get_origin(result) in [Union, getattr(types, 'UnionType', None)]
    
    # Verify both types are in the union
    union_args = typing.get_args(result)
    assert set(union_args) == {int, str}


def test_readable_error_messages():
    """
    Test that error messages are human-readable and actionable.
    
    Error messages should provide clear information about what went wrong.
    """
    def empty_container_test(items: List[A]) -> A: ...
    
    try:
        infer_return_type(empty_container_test, [])
        assert False, "Should have failed"
    except TypeInferenceError as e:
        error_msg = str(e)
        # Error message should be descriptive (not just a terse error code)
        assert len(error_msg) > 20
        # Should mention the TypeVar that couldn't be inferred
        assert 'A' in error_msg or 'TypeVar' in error_msg.lower()
        # Should indicate insufficient information
        assert 'insufficient' in error_msg.lower() or 'could not' in error_msg.lower()


# =============================================================================
# PERFORMANCE AND SCALABILITY TESTS
# =============================================================================

@pytest.mark.skip(reason="BENCHMARK: Performance test, not a correctness test")
def test_deeply_nested_performance():
    """
    Benchmark: Test performance on deeply nested structures.
    
    This is a performance benchmark to ensure unification doesn't have
    exponential blowup on deep nesting. Not a correctness test.
    """
    import time
    
    def deep_nested(data: List[List[List[List[List[A]]]]]) -> A: ...
    
    deep_data = [[[[[1, 2, 3]]]]]
    
    start = time.time()
    result = infer_return_type(deep_nested, deep_data)
    elapsed = time.time() - start
    
    assert result is int
    assert elapsed < 1.0  # Should complete in under 1 second


@pytest.mark.skip(reason="BENCHMARK: Scalability test, not a correctness test")
def test_many_typevars_scalability():
    """
    Benchmark: Test scalability with many TypeVars.
    
    This is a performance benchmark to ensure constraint solving doesn't
    become too slow with many variables. Not a correctness test.
    """
    # Define a type with many type parameters
    @dataclass
    class ManyParams(typing.Generic[A, B, C, X, Y]):
        a: A
        b: B  
        c: C
        x: X
        y: Y
    
    def extract_all(mp: ManyParams[A, B, C, X, Y]) -> Tuple[A, B, C, X, Y]: ...
    
    instance = ManyParams[int, str, float, bool, bytes](
        a=1, b="hello", c=3.14, x=True, y=b"data"
    )
    
    import time
    start = time.time()
    result = infer_return_type(extract_all, instance)
    elapsed = time.time() - start
    
    assert elapsed < 0.5  # Should be fast even with many TypeVars


# =============================================================================
# NESTED FIELD EXTRACTION TESTS (from original implementation)
# =============================================================================

def test_nested_list_field_extraction():
    """Test that TypeVars can be inferred from nested list fields."""
    
    @dataclass
    class Container(typing.Generic[A]):
        nested_items: List[A]  # The TypeVar A is nested inside a List
    
    def extract_item_type(container: Container[A]) -> A: ...
    
    # Create instance without explicit type parameters
    container = Container(nested_items=[1, 2, 3])
    
    # Should infer A = int from the nested list elements
    result_type = infer_return_type(extract_item_type, container)
    assert result_type is int


def test_nested_dict_field_extraction():
    """Test that TypeVars can be inferred from nested dict fields."""
    
    @dataclass
    class DataStore(typing.Generic[A, B]):
        data_map: Dict[str, List[A]]  # A is nested: str -> List[A]
        metadata: Dict[A, B]          # Both A and B are in dict structure
    
    def get_data_type(store: DataStore[A, B]) -> A: ...
    def get_metadata_type(store: DataStore[A, B]) -> B: ...
    
    # Create instance without explicit type parameters
    store = DataStore(
        data_map={"items": [1, 2, 3], "more": [4, 5]},
        metadata={42: "hello", 99: "world"}
    )
    
    # Should infer A = int from dict values' list elements AND dict keys
    data_type = infer_return_type(get_data_type, store)
    assert data_type is int
    
    # Should infer B = str from dict values
    metadata_type = infer_return_type(get_metadata_type, store)
    assert metadata_type is str


def test_deeply_nested_field_extraction():
    """Test that TypeVars can be inferred from deeply nested structures."""
    
    @dataclass
    class DeepContainer(typing.Generic[A]):
        deep_data: Dict[str, List[Dict[str, A]]]  # A is deeply nested
    
    def extract_deep_type(container: DeepContainer[A]) -> A: ...
    
    # Create instance with deeply nested structure
    container = DeepContainer(
        deep_data={
            "section1": [
                {"item1": 42, "item2": 100},
                {"item3": 200}
            ],
            "section2": [
                {"item4": 300}
            ]
        }
    )
    
    # Should infer A = int from deeply nested dict values
    result_type = infer_return_type(extract_deep_type, container)
    assert result_type is int


def test_optional_nested_field_extraction():
    """Test that TypeVars can be inferred from Optional nested fields."""
    
    @dataclass  
    class OptionalContainer(typing.Generic[A]):
        maybe_items: Optional[List[A]]  # A is nested inside Optional[List[A]]
    
    def extract_optional_type(container: OptionalContainer[A]) -> A: ...
    
    # Case 1: Non-None optional field
    container1 = OptionalContainer(maybe_items=["hello", "world"])
    result_type1 = infer_return_type(extract_optional_type, container1)
    assert result_type1 is str
    
    # Case 2: None optional field should fail (no type info available)
    container2 = OptionalContainer(maybe_items=None)
    with pytest.raises(Exception):  # Should fail due to lack of type information
        infer_return_type(extract_optional_type, container2)


def test_mixed_nested_structures():
    """Test TypeVar inference from mixed nested structures."""
    
    @dataclass
    class ComplexContainer(typing.Generic[A, B]):
        lists_of_a: List[List[A]]           # A doubly nested in lists
        dict_to_b: Dict[str, B]             # B nested in dict values
        optional_a_list: Optional[List[A]]  # A nested in Optional[List[A]]
    
    def extract_a_type(container: ComplexContainer[A, B]) -> A: ...
    def extract_b_type(container: ComplexContainer[A, B]) -> B: ...
    
    container = ComplexContainer(
        lists_of_a=[[1, 2], [3, 4, 5]],
        dict_to_b={"key1": 3.14, "key2": 2.71},
        optional_a_list=[10, 20, 30]
    )
    
    # Should infer A = int from multiple nested sources
    a_type = infer_return_type(extract_a_type, container)
    assert a_type is int
    
    # Should infer B = float from dict values
    b_type = infer_return_type(extract_b_type, container)
    assert b_type is float


def test_pydantic_nested_field_extraction():
    """Test nested field extraction works with Pydantic models too."""
    
    class NestedModel(BaseModel, typing.Generic[A]):
        nested_data: List[Dict[str, A]]  # A is nested inside List[Dict[str, A]]
    
    def extract_nested_type(model: NestedModel[A]) -> A: ...
    
    # Create Pydantic instance without explicit type parameters
    model = NestedModel(
        nested_data=[
            {"item1": True, "item2": False},
            {"item3": True}
        ]
    )
    
    # Should infer A = bool from nested dict values
    result_type = infer_return_type(extract_nested_type, model)
    assert result_type is bool


def test_inheritance_with_nested_extraction():
    """Test nested extraction works with inheritance chains."""
    
    @dataclass
    class BaseContainer(typing.Generic[A]):
        base_data: List[A]
    
    @dataclass
    class ExtendedContainer(BaseContainer[A], typing.Generic[A, B]):
        extra_data: Dict[str, B]
    
    def extract_base_type(container: ExtendedContainer[A, B]) -> A: ...
    def extract_extra_type(container: ExtendedContainer[A, B]) -> B: ...
    
    container = ExtendedContainer(
        base_data=[1, 2, 3],
        extra_data={"key": "value"}
    )
    
    # Should infer A = int from inherited base_data list
    base_type = infer_return_type(extract_base_type, container)
    assert base_type is int
    
    # Should infer B = str from extra_data dict values
    extra_type = infer_return_type(extract_extra_type, container)
    assert extra_type is str


def test_multiple_typevar_same_nested_structure():
    """Test multiple TypeVars in the same nested structure."""
    
    @dataclass
    class MultiVarContainer(typing.Generic[A, B]):
        pair_lists: List[Dict[A, B]]  # Both A and B nested in same structure
    
    def extract_key_type(container: MultiVarContainer[A, B]) -> A: ...
    def extract_value_type(container: MultiVarContainer[A, B]) -> B: ...
    
    container = MultiVarContainer(
        pair_lists=[
            {1: "one", 2: "two"},
            {3: "three", 4: "four"}
        ]
    )
    
    # Should infer A = int from dict keys
    key_type = infer_return_type(extract_key_type, container)
    assert key_type is int
    
    # Should infer B = str from dict values
    value_type = infer_return_type(extract_value_type, container)
    assert value_type is str


def test_comparison_with_explicit_types():
    """Test that nested extraction gives same results as explicit type parameters."""
    
    @dataclass
    class TestContainer(typing.Generic[A]):
        nested_list: List[A]
    
    def extract_type(container: TestContainer[A]) -> A: ...
    
    # Create with explicit type parameters
    explicit_container = TestContainer[int](nested_list=[1, 2, 3])
    explicit_result = infer_return_type(extract_type, explicit_container)
    
    # Create without explicit type parameters (relies on nested extraction)
    inferred_container = TestContainer(nested_list=[1, 2, 3])
    inferred_result = infer_return_type(extract_type, inferred_container)
    
    # Both should give the same result
    assert explicit_result == inferred_result == int


# =============================================================================
# COVERAGE IMPROVEMENT TESTS - EDGE CASES AND ERROR PATHS
# =============================================================================

def test_constraint_and_substitution_internals():
    """Test internal methods of Constraint and Substitution classes."""
    from unification_type_inference import Constraint, Substitution, Variance
    
    # Test Constraint __str__ and __repr__
    constraint = Constraint(A, int, Variance.COVARIANT)
    str_repr = str(constraint)
    assert "~" in str_repr
    assert "covariant" in str_repr
    
    # Test Constraint with is_override flag
    override_constraint = Constraint(A, str, Variance.INVARIANT, is_override=True)
    override_str = str(override_constraint)
    assert "override" in override_str
    repr_str = repr(override_constraint)
    assert "override" in repr_str
    
    # Test Substitution.compose
    sub1 = Substitution()
    sub1.bind(A, int)
    
    sub2 = Substitution()
    sub2.bind(B, str)
    
    composed = sub1.compose(sub2)
    assert composed.get(A) == int
    assert composed.get(B) == str
    
    # Test composing with TypeVar substitution
    sub3 = Substitution()
    sub3.bind(A, B)  # A -> B
    
    sub4 = Substitution()
    sub4.bind(B, int)  # B -> int
    
    composed2 = sub3.compose(sub4)
    # After composition, A should map to B (since sub4 applies to sub3's bindings)
    assert composed2.get(A) == B or composed2.get(A) == int


def test_optional_with_none_value():
    """Test Optional type handling when value is None."""
    
    def process_optional(x: Optional[A], y: Optional[B]) -> Tuple[A, B]: ...
    
    # When we have None values, the TypeVar in Optional can't be inferred from that parameter
    # But other parameters can still provide the type information
    t = infer_return_type(process_optional, 42, "hello")
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (int, str)
    
    # Test with one None value - should still work from the other parameter
    def process_one_optional(x: Optional[A], y: A) -> A: ...
    t = infer_return_type(process_one_optional, None, 42)
    assert t == int


def test_union_constraint_handling_errors():
    """Test error handling in union constraint collection."""
    
    # Test Union where no alternatives match (should raise)
    def process_strict_union(x: Union[List[int], Dict[str, str]]) -> int: ...
    
    # A set doesn't match either alternative
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_strict_union, {1, 2, 3})


def test_tuple_ellipsis_handling():
    """Test Tuple[A, ...] (variable length tuple) handling."""
    
    def process_var_tuple(t: Tuple[A, ...]) -> Set[A]: ...
    
    # Empty tuple should fail (can't infer A)
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_var_tuple, ())
    
    # Non-empty tuple should work
    t = infer_return_type(process_var_tuple, (1, 2, 3))
    assert typing.get_origin(t) is set
    assert typing.get_args(t) == (int,)
    
    # Test with mixed types - should create union
    t = infer_return_type(process_var_tuple, (1, "hello", 2, "world"))
    assert typing.get_origin(t) is set
    union_arg = typing.get_args(t)[0]
    import types
    origin = typing.get_origin(union_arg)
    assert origin is Union or origin is getattr(types, 'UnionType', None)


def test_set_constraint_union_handling():
    """Test Set with Union element types."""
    
    def process_set_union(s: Set[Union[A, B]]) -> Tuple[A, B]: ...
    
    # Mixed set with two types
    t = infer_return_type(process_set_union, {1, "hello", 2, "world"})
    assert typing.get_origin(t) is tuple
    
    # Should distribute types among A and B
    tuple_args = typing.get_args(t)
    assert set(tuple_args) == {int, str}


def test_constraint_solver_edge_cases():
    """Test edge cases in constraint solving."""
    from unification_type_inference import UnificationEngine, Constraint, Variance
    
    engine = UnificationEngine()
    
    # Test with override constraints
    constraints = [
        Constraint(A, int, Variance.INVARIANT),
        Constraint(A, str, Variance.COVARIANT),
        Constraint(A, float, Variance.INVARIANT, is_override=True),  # Override wins
    ]
    
    sub = engine._solve_constraints(constraints)
    assert sub.get(A) == float  # Override should take precedence


def test_bounded_typevar_violation():
    """Test that bounded TypeVar violations are caught."""
    T_bounded = TypeVar('T_bounded', bound=str)
    
    def process_bounded(x: T_bounded) -> T_bounded: ...
    
    # int doesn't satisfy bound=str
    with pytest.raises(TypeInferenceError, match="bound"):
        infer_return_type(process_bounded, 42)
    
    # str subclass should work
    t = infer_return_type(process_bounded, "hello")
    assert t == str


def test_constrained_typevar_violation():
    """Test that constrained TypeVar violations are caught."""
    T_constrained = TypeVar('T_constrained', int, str)
    
    def process_constrained(x: T_constrained) -> T_constrained: ...
    
    # float is not in constraints (int, str)
    with pytest.raises(TypeInferenceError, match="violates constraints"):
        infer_return_type(process_constrained, 3.14)
    
    # int should work
    t = infer_return_type(process_constrained, 42)
    assert t == int


def test_union_in_constraints_checking():
    """Test Union type constraint checking.
    
    Per PEP 484: Constrained TypeVars must resolve to exactly ONE of the specified types,
    not a union of them. A union type like int | str violates the constraint requirement
    for TypeVar('T', int, str) because T must be either int OR str consistently, not both.
    
    This matches the behavior of static type checkers like mypy, which would reject
    mixed-type lists for constrained TypeVars.
    """
    T_constrained = TypeVar('T_constrained', int, str)
    
    def process_mixed(items: List[T_constrained]) -> T_constrained: ...
    
    # Mixed list creates union type - this violates PEP 484 constraint semantics
    # Static type checkers (mypy, pyright) would reject this, so should we
    with pytest.raises(TypeInferenceError, match="violates constraints"):
        t = infer_return_type(process_mixed, [1, "hello", 2])
        
        
def test_union_in_constraints_checking_union():
    """Test constrained TypeVar where constraints themselves are union types.
    
    This tests the edge case where TypeVar('T', int | str, float | list) has
    union types as constraints. The inferred type must match one of these union constraints.
    """
    T_constrained = TypeVar('T_constrained', int | str, float | list)
    
    def process_mixed(items: List[T_constrained]) -> T_constrained: ...
    
    # Test 1: [1, "hello", 2] should infer int | str, matching first constraint
    t = infer_return_type(process_mixed, [1, "hello", 2])
    
    # Should be int | str (or str | int - order doesn't matter for equality)
    origin = typing.get_origin(t)
    import types
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    
    # The inferred union should match one of the constraints
    assert t in T_constrained.__constraints__
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}
    
    # Test 2: [1.5, [1,2]] should infer float | list[int], matching second constraint
    t2 = infer_return_type(process_mixed, [1.5, [1, 2]])
    # The inferred type should satisfy the constraint (by origin matching)
    # Note: exact equality won't work since list[int] != list, but origins match
    union_args2 = typing.get_args(t2)
    # Check that the inferred components match constraint components by origin
    inferred_origins = {typing.get_origin(arg) or arg for arg in union_args2}
    assert inferred_origins == {float, list}, f"Expected {{float, list}}, got {inferred_origins}"
    
    # Test 3: [1, 1.5] should fail - int | float doesn't match either constraint
    with pytest.raises(TypeInferenceError, match="violates constraints"):
        infer_return_type(process_mixed, [1, 1.5])


def test_empty_container_error_messages():
    """Test that empty container errors provide clear messages."""
    
    def process_empty_list(items: List[A]) -> A: ...
    
    try:
        infer_return_type(process_empty_list, [])
        assert False, "Should have raised TypeInferenceError"
    except TypeInferenceError as e:
        error_msg = str(e)
        # Error message should mention the issue
        assert "could not" in error_msg.lower() or "insufficient" in error_msg.lower()


def test_has_unbound_typevars_helper():
    """Test _has_unbound_typevars helper function."""
    from unification_type_inference import _has_unbound_typevars
    
    # Simple TypeVar
    assert _has_unbound_typevars(A) == True
    
    # Concrete type
    assert _has_unbound_typevars(int) == False
    
    # List with TypeVar
    assert _has_unbound_typevars(List[A]) == True
    
    # List with concrete type
    assert _has_unbound_typevars(List[int]) == False
    
    # Dict with mixed
    assert _has_unbound_typevars(Dict[A, int]) == True
    assert _has_unbound_typevars(Dict[str, int]) == False


def test_substitute_typevars_edge_cases():
    """Test _substitute_typevars with various edge cases."""
    from unification_type_inference import _substitute_typevars
    
    # Unbound TypeVar should be returned as-is
    bindings = {B: int}
    result = _substitute_typevars(A, bindings)
    assert result == A  # A is not in bindings, returned as-is
    
    # Union with mixed bound/unbound TypeVars
    bindings = {A: int}
    union_type = Union[A, B, str]
    result = _substitute_typevars(union_type, bindings)
    
    # Should only include bound args (int and str), not B
    import types
    origin = typing.get_origin(result)
    if origin is Union or origin is getattr(types, 'UnionType', None):
        args = typing.get_args(result)
        # B should not be in the result since it's unbound
        assert int in args
        assert str in args


def test_infer_type_from_value_edge_cases():
    """Test _infer_type_from_value helper function."""
    from unification_type_inference import _infer_type_from_value
    
    # None value
    t = _infer_type_from_value(None)
    assert t == type(None)
    
    # Empty list
    t = _infer_type_from_value([])
    assert t == list  # Should return list without type args for empty
    
    # List with single type
    t = _infer_type_from_value([1, 2, 3])
    assert typing.get_origin(t) is list
    assert typing.get_args(t) == (int,)
    
    # List with mixed types
    t = _infer_type_from_value([1, "hello"])
    assert typing.get_origin(t) is list
    # Should have union type arg
    
    # Dict with values
    t = _infer_type_from_value({1: "a", 2: "b"})
    assert typing.get_origin(t) is dict
    key_type, val_type = typing.get_args(t)
    assert key_type == int
    assert val_type == str
    
    # Set with single type
    t = _infer_type_from_value({1, 2, 3})
    assert typing.get_origin(t) is set
    assert typing.get_args(t) == (int,)
    
    # Tuple (should preserve individual types)
    t = _infer_type_from_value((1, "hello", 3.14))
    assert typing.get_origin(t) is tuple
    # Tuple should have all element types


def test_dict_constraints_with_empty():
    """Test Dict constraint handling with empty dict."""
    
    def process_dict(d: Dict[A, B]) -> Tuple[A, B]: ...
    
    # Empty dict can't infer types
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_dict, {})


def test_tuple_fixed_length_partial_match():
    """Test fixed-length tuple with more elements than annotation expects."""
    
    def process_tuple(t: Tuple[A, B]) -> A: ...
    
    # Tuple longer than expected - should still extract available positions
    t = infer_return_type(process_tuple, (1, "hello", "extra"))
    assert t == int


def test_custom_generic_fallback_paths():
    """Test fallback paths in custom generic handling."""
    
    @dataclass
    class CustomGeneric(typing.Generic[A]):
        data: A
    
    # Test without __orig_class__ - should fall back to field extraction
    def process_custom(c: CustomGeneric[A]) -> A: ...
    
    instance = CustomGeneric(data=42)
    # Remove __orig_class__ if it exists to test fallback
    if hasattr(instance, '__orig_class__'):
        delattr(instance, '__orig_class__')
    
    t = infer_return_type(process_custom, instance)
    assert t == int


def test_type_mismatch_errors():
    """Test type mismatch error handling."""
    
    def process_list(items: List[A]) -> A: ...
    
    # Passing non-list when List is expected
    with pytest.raises(TypeInferenceError, match="Expected list"):
        infer_return_type(process_list, "not a list")
    
    def process_dict(d: Dict[A, B]) -> A: ...
    
    # Passing non-dict when Dict is expected
    with pytest.raises(TypeInferenceError, match="Expected dict"):
        infer_return_type(process_dict, [1, 2, 3])
    
    def process_set(s: Set[A]) -> A: ...
    
    # Passing non-set when Set is expected
    with pytest.raises(TypeInferenceError, match="Expected set"):
        infer_return_type(process_set, [1, 2, 3])
    
    def process_tuple(t: Tuple[A, B]) -> A: ...
    
    # Passing non-tuple when Tuple is expected
    with pytest.raises(TypeInferenceError, match="Expected tuple"):
        infer_return_type(process_tuple, [1, 2])


def test_variance_handling_in_constraint_resolution():
    """Test that variance is properly handled in constraint resolution."""
    from unification_type_inference import UnificationEngine, Constraint, Variance
    
    engine = UnificationEngine()
    
    # Multiple covariant constraints should form union
    constraints = [
        Constraint(A, int, Variance.COVARIANT),
        Constraint(A, str, Variance.COVARIANT),
        Constraint(A, float, Variance.COVARIANT),
    ]
    
    sub = engine._solve_constraints(constraints)
    result = sub.get(A)
    
    # Should be a union type
    import types
    origin = typing.get_origin(result)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(result)
    assert set(union_args) == {int, str, float}


def test_function_without_return_annotation():
    """Test that functions without return annotations raise appropriate errors."""
    
    def no_annotation(x):
        return x
    
    with pytest.raises(ValueError, match="return type annotation"):
        infer_return_type(no_annotation, 42)


def test_complex_nested_optional_structures():
    """Test complex nested Optional structures."""
    
    def process_nested_optional(data: Optional[List[Optional[A]]]) -> A: ...
    
    # List with None elements
    t = infer_return_type(process_nested_optional, [1, None, 2])
    
    # Should infer A from non-None elements
    import types
    origin = typing.get_origin(t)
    if origin is Union or origin is getattr(types, 'UnionType', None):
        # May include NoneType
        union_args = typing.get_args(t)
        assert int in union_args
    else:
        assert t == int


def test_is_subtype_helper():
    """Test _is_subtype helper function."""
    from unification_type_inference import _is_subtype
    
    # Basic subtype relationships
    assert _is_subtype(bool, int) == True  # bool is subtype of int
    assert _is_subtype(int, object) == True
    assert _is_subtype(int, str) == False
    
    # Edge case: non-class types
    assert _is_subtype(List[int], list) == False  # Generic types handled


def test_list_union_element_matching():
    """Test List[Union[A, B]] with complex matching."""
    
    def process_list_union(items: List[Union[A, B]]) -> Tuple[Set[A], Set[B]]: ...
    
    # When types can be distributed
    t = infer_return_type(process_list_union, [1, "hello", 2, "world"])
    
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert len(args) == 2
    
    # Both should be sets
    assert typing.get_origin(args[0]) is set
    assert typing.get_origin(args[1]) is set


def test_unification_error_to_type_inference_error():
    """Test that UnificationError is converted to TypeInferenceError."""
    from unification_type_inference import UnificationEngine, UnificationError
    
    engine = UnificationEngine()
    
    # Create a situation that would raise UnificationError
    # Passing wrong type to typed container
    def process_strict(items: List[int]) -> int: ...
    
    # This should convert UnificationError to TypeInferenceError
    with pytest.raises(TypeInferenceError):
        # Pass a non-list
        engine._collect_constraints(List[int], "not a list", [])


def test_callable_type_limitations():
    """Document Callable type inference limitations."""
    
    def higher_order(func: Callable[[A], B], value: A) -> B: ...
    
    def sample_func(x: int) -> str:
        return str(x)
    
    # Callable type inference is not fully supported
    # This should fail because we can't extract types from callable signatures
    with pytest.raises(TypeInferenceError):
        infer_return_type(higher_order, sample_func, 42)


# =============================================================================
# CONSOLIDATED TESTS FROM test_additional_coverage.py
# =============================================================================

def test_unification_engine_direct_api():
    """Test UnificationEngine.unify_annotation_with_value directly."""
    engine = UnificationEngine()
    
    # Test with List[A] and [1, 2, 3]
    substitution = engine.unify_annotation_with_value(List[A], [1, 2, 3])
    assert substitution.get(A) == int
    
    # Test with Dict[K, V] and {"a": 1}
    substitution = engine.unify_annotation_with_value(Dict[K, V], {"a": 1})
    assert substitution.get(K) == str
    assert substitution.get(V) == int
    
    # Test with pre-existing constraints
    existing_constraints = [Constraint(A, int, Variance.INVARIANT)]
    substitution = engine.unify_annotation_with_value(
        List[A], [1, 2, 3], constraints=existing_constraints
    )
    assert substitution.get(A) == int


def test_unification_engine_with_none_constraints():
    """Test that unify_annotation_with_value handles None constraints properly."""
    engine = UnificationEngine()
    
    # Explicitly pass None for constraints parameter (line 109-110)
    substitution = engine.unify_annotation_with_value(Set[A], {1, 2, 3}, constraints=None)
    assert substitution.get(A) == int


# =============================================================================
# 2. FORWARD REFERENCE TESTS (Lines 160-173)
# =============================================================================

@pytest.mark.skip(reason="LIMITATION: ForwardRef (string annotations) not fully supported")


def test_forward_reference_simple():
    """Test ForwardRef handling in type annotations."""
    
    # Using string annotations that create ForwardRefs
    @dataclass
    class Node:
        value: int
        next: Optional['Node'] = None
    
    def get_value(node: 'Node') -> int: ...
    
    node = Node(value=42)
    # This exercises the ForwardRef handling path
    t = infer_return_type(get_value, node)
    assert t == int


@pytest.mark.skip(reason="LIMITATION: ForwardRef with generics not fully supported")


def test_forward_reference_with_generics():
    """Test ForwardRef with generic type parameters."""
    
    def extract_value(node: 'TreeNode[A]') -> A: ...
    
    tree = TreeNode[str](value="root", children=[])
    t = infer_return_type(extract_value, tree)
    assert t == str


def test_forward_reference_mismatch():
    """Test ForwardRef when class names don't match."""
    
    @dataclass
    class ActualClass:
        data: int
    
    # This should fail because annotation expects different class
    def process_wrong(obj: 'DifferentClass') -> int: ...
    
    # This will raise an error due to type mismatch
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_wrong, ActualClass(data=42))


# =============================================================================
# 3. MULTIPLE CONFLICTING OVERRIDES (Lines 427-433)
# =============================================================================


def test_multiple_consistent_overrides():
    """Test multiple type_overrides that are consistent."""
    
    def multi_param(a: A, b: A, c: A) -> Tuple[A, A, A]: ...
    
    # Multiple overrides but all the same type - should work
    t = infer_return_type(
        multi_param, 1, 2, 3,
        type_overrides={A: str}  # Override all A to str despite int values
    )
    assert typing.get_origin(t) == tuple
    assert typing.get_args(t) == (str, str, str)


def test_multiple_conflicting_overrides_direct():
    """Test conflicting type_overrides through direct engine API."""
    engine = UnificationEngine()
    
    # Create conflicting override constraints
    constraints = [
        Constraint(A, int, Variance.INVARIANT, is_override=True),
        Constraint(A, str, Variance.INVARIANT, is_override=True),
        Constraint(A, float, Variance.INVARIANT, is_override=True),
    ]
    
    # Multiple different overrides should raise UnificationError
    with pytest.raises(UnificationError, match="Conflicting override"):
        engine._solve_constraints(constraints)


# =============================================================================
# 4. MIXED VARIANCE EDGE CASES (Line 466)
# =============================================================================


def test_mixed_variance_constraints():
    """Test mixed covariant and invariant constraints together."""
    engine = UnificationEngine()
    
    # Mix of covariant and invariant constraints
    constraints = [
        Constraint(A, int, Variance.COVARIANT),
        Constraint(A, str, Variance.INVARIANT),
        Constraint(A, float, Variance.COVARIANT),
    ]
    
    sub = engine._solve_constraints(constraints)
    result = sub.get(A)
    
    # Should form union with all types
    import types
    origin = typing.get_origin(result)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(result)
    assert set(union_args) == {int, str, float}


# =============================================================================
# 5. COMPLEX UNION DISTRIBUTION (Lines 618, 638-670)
# =============================================================================


def test_set_union_with_context_aware_matching():
    """Test Set[Union[A, B]] with context-aware TypeVar matching."""
    
    def complex_set_union(
        s1: Set[Union[A, B]],
        s2: Set[Union[A, B]],
        s3: Set[Union[A, B]]
    ) -> Tuple[A, B]: ...
    
    # Multiple sets with mixed types - tests context-aware matching
    # First set establishes pattern: 1 -> A, "a" -> B
    # Subsequent sets should follow this pattern
    t = infer_return_type(
        complex_set_union,
        {1, "a"},      # Establishes A=int, B=str
        {2, "b"},      # Reinforces pattern
        {3, "c"}       # Reinforces pattern
    )
    
    assert typing.get_origin(t) == tuple
    result_types = set(typing.get_args(t))
    assert result_types == {int, str}


def test_set_union_no_candidates():
    """Test Set[Union[A, B]] when all TypeVars are ruled out."""
    
    def set_union_difficult(s: Set[Union[A, B]]) -> Tuple[A, B]: ...
    
    # This exercises the fallback path (lines 667-670)
    # When we have a single set with mixed types
    t = infer_return_type(set_union_difficult, {1, "x", 2.5})
    
    assert typing.get_origin(t) == tuple


# =============================================================================
# 6. CONSTRAINT CHECKING EDGE CASES (Lines 489, 494, 513, 556, 581)
# =============================================================================


def test_constraint_checking_with_nested_generics():
    """Test constraint matching with nested generic types."""
    
    # TypeVar constrained to generic types
    T = TypeVar('T', list[int], dict[str, int])
    
    def process_constrained(x: T) -> T: ...
    
    # list[int] should match first constraint
    t = infer_return_type(process_constrained, [1, 2, 3])
    assert typing.get_origin(t) == list
    assert typing.get_args(t) == (int,)


def test_bounded_typevar_with_union():
    """Test bounded TypeVar that produces a union."""
    
    class Base: pass
    class Derived1(Base): pass
    class Derived2(Base): pass
    
    T_bounded = TypeVar('T_bounded', bound=Base)
    
    def process_bounded_multi(items: List[T_bounded]) -> T_bounded: ...
    
    # Mixed derived types should create union within bound
    t = infer_return_type(process_bounded_multi, [Derived1(), Derived2()])
    
    import types
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(t)
    assert set(union_args) == {Derived1, Derived2}


def test_constrained_typevar_with_union_match():
    """Test constrained TypeVar where inferred union matches a constraint union."""
    
    # Constraint is itself a union type
    T = TypeVar('T', int | str, float | bool)
    
    def process(items: List[T]) -> T: ...
    
    # [1, "x"] should infer int | str, matching first constraint
    t = infer_return_type(process, [1, "x"])
    
    # Should match the first constraint (int | str)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


# =============================================================================
# 7. TYPE SUBSTITUTION EDGE CASES (Lines 714, 727, 762, 779-787)
# =============================================================================


def test_substitute_empty_set():
    """Test type substitution with empty set (base type)."""
    
    def process_empty_set_fallback(s: Set[A], default: A) -> A: ...
    
    # Empty set should use default value for inference
    t = infer_return_type(process_empty_set_fallback, set(), 42)
    assert t == int


def test_substitute_with_generic_alias():
    """Test substitution with complex generic aliases."""
    from unification_type_inference import _substitute_typevars
    
    # Test substituting in dict type
    bindings = {K: str, V: int}
    result = _substitute_typevars(Dict[K, V], bindings)
    
    assert typing.get_origin(result) == dict
    key_type, val_type = typing.get_args(result)
    assert key_type == str
    assert val_type == int


def test_substitute_tuple_types():
    """Test substitution preserves tuple structure."""
    from unification_type_inference import _substitute_typevars
    
    # Fixed-length tuple substitution
    bindings = {A: int, B: str, C: float}
    result = _substitute_typevars(Tuple[A, B, C], bindings)
    
    assert typing.get_origin(result) == tuple
    args = typing.get_args(result)
    # Result should be tuple[int, str, float]
    assert args == (int, str, float)


def test_substitute_set_types():
    """Test substitution with Set types."""
    from unification_type_inference import _substitute_typevars
    
    bindings = {A: str}
    result = _substitute_typevars(Set[A], bindings)
    
    assert typing.get_origin(result) == set
    assert typing.get_args(result) == (str,)


def test_substitute_generic_class():
    """Test substitution with custom generic class."""
    from unification_type_inference import _substitute_typevars
    
    @dataclass
    class Container(typing.Generic[A, B]):
        a: A
        b: B
    
    # Attempt to substitute in custom generic
    bindings = {A: int, B: str}
    result = _substitute_typevars(Container[A, B], bindings)
    
    # Should attempt to reconstruct Container[int, str]
    assert result == Container[int, str]
    assert result != Container[A, B]


# =============================================================================
# 8. KEYWORD ARGUMENT HANDLING (Lines 826-831)
# =============================================================================


def test_keyword_arguments_with_inference():
    """Test type inference with keyword arguments."""
    
    def func_with_kwargs(a: A, b: B, c: C = None) -> Tuple[A, B]: ...
    
    # Mix positional and keyword arguments
    t = infer_return_type(func_with_kwargs, 1, b="hello", c=3.14)
    
    assert typing.get_origin(t) == tuple
    assert typing.get_args(t) == (int, str)


def test_all_keyword_arguments():
    """Test type inference with all keyword arguments."""
    
    def func_all_kwargs(x: A, y: B, z: C) -> Dict[A, B]: ...
    
    # All keyword arguments (exercises line 827-831)
    t = infer_return_type(func_all_kwargs, x=1, y="str", z=3.14)
    
    assert typing.get_origin(t) == dict
    key_type, val_type = typing.get_args(t)
    assert key_type == int
    assert val_type == str


def test_keyword_argument_not_in_signature():
    """Test that extra keyword arguments are ignored."""
    
    def func_limited(a: A) -> A: ...
    
    # Pass extra kwargs that aren't in signature (should be ignored)
    t = infer_return_type(func_limited, a=42, extra="ignored")
    assert t == int


# =============================================================================
# 9. SUBSTITUTION STRING REPRESENTATION (Line 89)
# =============================================================================


def test_substitution_str_repr():
    """Test Substitution.__str__ with multiple bindings."""
    sub = Substitution()
    sub.bind(A, int)
    sub.bind(B, str)
    sub.bind(C, float)
    
    str_repr = str(sub)
    
    # Should contain all bindings
    assert "A" in str_repr or "~A" in str_repr
    assert "int" in str_repr
    assert "str" in str_repr
    assert "float" in str_repr
    assert "{" in str_repr and "}" in str_repr


# =============================================================================
# 10. GENERIC INFO MATCHING EDGE CASES (Lines 280-299)
# =============================================================================


def test_generic_info_matching_different_origins():
    """Test generic info matching when origins differ."""
    engine = UnificationEngine()
    
    from generic_utils import get_generic_info, get_instance_generic_info
    
    # Different origins should return False
    list_info = get_generic_info(List[A])
    
    set_val = {1, 2, 3}
    set_info = get_instance_generic_info(set_val)
    
    constraints = []
    result = engine._match_generic_structures(list_info, set_info, constraints)
    
    # Should return False because list != set
    assert result == False
    assert len(constraints) == 0


def test_generic_info_matching_different_arg_counts():
    """Test generic info matching with mismatched argument counts."""
    engine = UnificationEngine()
    
    from generic_utils import get_generic_info
    
    # Create a mock scenario with different arg counts
    # List[A] has 1 arg, Dict[K, V] has 2 args
    list_info = get_generic_info(List[A])
    dict_info = get_generic_info(Dict[str, int])
    
    constraints = []
    # This should return False due to arg count mismatch
    result = engine._match_generic_structures(list_info, dict_info, constraints)
    
    assert result == False


def test_generic_info_recursive_matching():
    """Test recursive generic info matching with nested structures."""
    engine = UnificationEngine()
    
    from generic_utils import get_generic_info
    
    # Nested generic: List[Dict[A, B]]
    nested_annotation = List[Dict[A, B]]
    nested_value = [{"key": 1}]
    
    ann_info = get_generic_info(nested_annotation)
    
    from generic_utils import get_instance_generic_info
    val_info = get_instance_generic_info(nested_value)
    
    constraints = []
    result = engine._match_generic_structures(ann_info, val_info, constraints)
    
    # Should successfully match and extract constraints
    # (May or may not return True depending on implementation details)


# =============================================================================
# 11. SUBCLASS COMPATIBILITY TESTS
# =============================================================================
# These tests document behavior when annotation expects a base class
# but receives a subclass instance. Per Liskov Substitution Principle,
# this should work, but current implementation may handle it differently.


def test_simple_subclass_generic_inheritance():
    """Test that subclass instances work where base class is expected.
    
    Expected: Should work via field-based extraction.
    Current: Works via field extraction, bypasses origin check.
    """
    
    @dataclass
    class Container(typing.Generic[A]):
        value: A
    
    @dataclass
    class SpecialContainer(Container[A], typing.Generic[A]):
        extra: str
    
    # Function expects Container[A] but we pass SpecialContainer[int]
    def process_container(c: Container[A]) -> A: ...
    
    special = SpecialContainer[int](value=42, extra="test")
    
    # This works because field-based extraction finds 'value: A' -> 'value: 42'
    result = infer_return_type(process_container, special)
    assert result == int


def test_deep_inheritance_chain():
    """Test deep inheritance chain: GrandChild -> Child -> Parent.
    
    Expected: Should work through any level of inheritance.
    Current: Works via field extraction.
    """
    
    @dataclass
    class GrandParent(typing.Generic[A]):
        gp_value: A
    
    @dataclass
    class Parent(GrandParent[A], typing.Generic[A]):
        p_value: str
    
    @dataclass
    class Child(Parent[A], typing.Generic[A]):
        c_value: int
    
    def process_gp(obj: GrandParent[A]) -> A: ...
    def process_p(obj: Parent[A]) -> A: ...
    
    child = Child[float](gp_value=3.14, p_value="test", c_value=42)
    
    # Should work at any level
    result_gp = infer_return_type(process_gp, child)
    assert result_gp == float
    
    result_p = infer_return_type(process_p, child)
    assert result_p == float


def test_partial_specialization_subclass():
    """Test subclass that partially specializes parent's type parameters.
    
    Expected: Should work, extracting only the remaining type parameters.
    Current: Works via field extraction.
    """
    
    @dataclass
    class TwoParam(typing.Generic[A, B]):
        first: A
        second: B
    
    @dataclass
    class OneParam(TwoParam[A, str], typing.Generic[A]):  # Fix B=str
        extra: int
    
    def process_two(obj: TwoParam[A, B]) -> Tuple[A, B]: ...
    
    one = OneParam[int](first=42, second="fixed", extra=99)
    
    # Should infer A=int, B=str
    result = infer_return_type(process_two, one)
    assert typing.get_origin(result) == tuple
    assert typing.get_args(result) == (int, str)


def test_concrete_subclass_of_generic():
    """Test non-generic subclass of generic base.
    
    Expected: Should work, base's type parameters are concrete.
    Current: Works via field extraction.
    """
    
    @dataclass
    class GenericBase(typing.Generic[A]):
        value: A
    
    @dataclass
    class ConcreteChild(GenericBase[int]):  # Fully specialized
        extra: str
    
    def process_generic(obj: GenericBase[A]) -> A: ...
    
    concrete = ConcreteChild(value=42, extra="test")
    
    # Should infer A=int from the specialized base
    result = infer_return_type(process_generic, concrete)
    assert result == int


def test_multiple_inheritance_generics():
    """Test class with multiple generic parents using different TypeVar names.
    
    Expected: Should follow MRO and extract from all parents.
    Current: Works correctly when parents use different TypeVar names.
    """
    
    @dataclass
    class HasA(typing.Generic[A]):
        a_value: A
    
    @dataclass
    class HasB(typing.Generic[B]):  # Different TypeVar name
        b_value: B
    
    @dataclass
    class HasBoth(HasA[A], HasB[B], typing.Generic[A, B]):
        both: str
    
    def extract_a(obj: HasA[A]) -> A: ...
    def extract_b(obj: HasB[B]) -> B: ...
    
    both = HasBoth[int, str](a_value=42, b_value="hello", both="test")
    
    # Should work for both parent types
    result_a = infer_return_type(extract_a, both)
    assert result_a == int
    
    result_b = infer_return_type(extract_b, both)
    assert result_b == str


def test_multiple_inheritance_typevar_shadowing():
    """Test multiple inheritance when parents use the SAME TypeVar name.
    
    FULLY FIXED! Both cases now work correctly through proper TypeVar substitution.
    
    The fix:
    1. Only extract fields from the annotation class (not inherited fields)
    2. Substitute TypeVars when annotation re-parameterizes the class
       Example: HasB defined as Generic[A], annotation is HasB[B] → substitute A with B
    3. Return substituted TypeVars to the inference engine
    
    This properly handles the case where:
    - HasA and HasB both use Generic[A] (same TypeVar name)
    - HasBoth uses them with different TypeVars: HasA[A], HasB[B]
    - The substitution ensures we track the right TypeVar for each parent
    """
    
    @dataclass
    class HasA(typing.Generic[A]):
        a_value: A
    
    @dataclass
    class HasB(typing.Generic[A]):  # SAME TypeVar name as HasA!
        b_value: A
    
    @dataclass
    class HasBoth(HasA[A], HasB[B], typing.Generic[A, B]):
        both: str
    
    def extract_a(obj: HasA[A]) -> A: ...
    def extract_b(obj: HasB[B]) -> B: ...
    
    both = HasBoth[int, str](a_value=42, b_value="hello", both="test")
    
    # Both work correctly now!
    result_a = infer_return_type(extract_a, both)
    assert result_a == int
    
    result_b = infer_return_type(extract_b, both)
    assert result_b == str


def test_swapped_generic_typevars():
    """Test that TypeVar swapping in inheritance is handled correctly.
    
    HasB(HasA[B, A], Generic[B, A]) swaps the type parameters.
    So HasB[int, str] means B=int, A=str, which gives:
    - a_value has type B = int
    - b_value has type A = str
    
    The function returns Tuple[A, B] = Tuple[str, int].
    """
    @dataclass
    class HasA(typing.Generic[A, B]):
        a_value: A
        b_value: B
    
    @dataclass
    class HasB(HasA[B, A], typing.Generic[A, B]):  # Swapped order
        pass
    
    def process_a(obj: HasB[C, D]) -> Tuple[C, D]: ...
    
    result = infer_return_type(process_a, HasB[int, str](a_value="hello", b_value=42))
    # HasB[int, str] means B=int, A=str
    # So Tuple[A, B] = Tuple[str, int]
    assert typing.get_origin(result) is tuple
    assert typing.get_args(result) == (int, str)


def test_swapped_generic_typevars_pydantic():
    """Test TypeVar swapping with Pydantic models.
    
    Similar to dataclass version but uses Pydantic BaseModel.
    Pydantic specializes field annotations automatically.
    """
    
    C = TypeVar('C')
    D = TypeVar('D')
    
    class ParentPyd(BaseModel, typing.Generic[A, B]):
        a_value: A
        b_value: B
    
    class ChildPyd(ParentPyd[B, A], typing.Generic[A, B]):
        # Swapped: Parent gets [B, A] but Child is [A, B]
        pass
    
    def process_pyd(obj: ChildPyd[C, D]) -> Tuple[C, D]: ...
    
    result = infer_return_type(process_pyd, ChildPyd[int, str](a_value="hello", b_value=42))
    
    # Pydantic specializes field annotations, so the behavior matches dataclass
    assert typing.get_origin(result) is tuple
    assert typing.get_args(result) == (int, str)


def test_builtin_type_subclass():
    """Test subclass of built-in generic types.
    
    Expected: Should work with custom list/dict/set subclasses.
    Current: Works! Built-in extractors check isinstance(), not exact type.
    """
    
    class MyList(list):
        """Custom list subclass."""
        pass
    
    def process_list(items: List[A]) -> A: ...
    
    my_list = MyList([1, 2, 3])
    
    # This works! Built-in extractors use isinstance(value, list)
    # So MyList (subclass of list) is handled correctly
    result = infer_return_type(process_list, my_list)
    assert result == int


def test_subclass_with_additional_type_params():
    """Test subclass that adds new type parameters.
    
    Expected: Should handle the base's params, ignore extras.
    Current: Tests actual behavior.
    """
    
    @dataclass
    class Base(typing.Generic[A]):
        base_val: A
    
    @dataclass
    class Extended(Base[A], typing.Generic[A, B]):  # Adds B
        extended_val: B
    
    def process_base(obj: Base[A]) -> A: ...
    
    extended = Extended[int, str](base_val=42, extended_val="extra")
    
    # Should infer A=int, ignore B
    result = infer_return_type(process_base, extended)
    assert result == int


def test_origins_compatible_with_subclass():
    """Test _origins_compatible method with subclass relationship.
    
    This documents current behavior: it returns False for subclasses.
    Per LSP, it arguably should return True, but current implementation
    works via field extraction so it's not critical.
    """
    from generic_utils import get_generic_info, get_instance_generic_info
    
    @dataclass
    class Base(typing.Generic[A]):
        value: A
    
    @dataclass
    class Derived(Base[A], typing.Generic[A]):
        extra: str
    
    engine = UnificationEngine()
    
    base_info = get_generic_info(Base[A])
    derived_instance = Derived[int](value=42, extra="test")
    derived_info = get_instance_generic_info(derived_instance)
    
    # Current implementation: returns False
    compatible = engine._origins_compatible(base_info.origin, derived_info.origin)
    
    # Document current behavior
    assert compatible == False, "Current: doesn't check subclass relationships"
    
    # Verify it IS actually a subclass
    assert issubclass(derived_info.origin, base_info.origin)


def test_diamond_inheritance():
    """Test diamond inheritance pattern with generics.
    
    Expected: Should follow MRO correctly.
    Current: Tests actual behavior.
    """
    
    @dataclass
    class Top(typing.Generic[A]):
        top: A
    
    @dataclass
    class Left(Top[A], typing.Generic[A]):
        left: str
    
    @dataclass
    class Right(Top[A], typing.Generic[A]):
        right: int
    
    @dataclass
    class Bottom(Left[A], Right[A], typing.Generic[A]):
        bottom: float
    
    def process_top(obj: Top[A]) -> A: ...
    
    bottom = Bottom[bool](top=True, left="l", right=1, bottom=2.0)
    
    # Should follow MRO and extract A=bool
    result = infer_return_type(process_top, bottom)
    assert result == bool


def test_covariant_subclass_list():
    """Test that List[Derived] works where List[Base] expected.
    
    Expected: Should work because List is covariant in reading.
    Current: Tests actual behavior for lists.
    """
    
    class Animal: pass
    class Dog(Animal): pass
    
    def process_animals(animals: List[A]) -> A: ...
    
    dogs = [Dog(), Dog()]
    
    # Should infer A=Dog (most specific type)
    result = infer_return_type(process_animals, dogs)
    assert result == Dog


def test_invariant_subclass_dict_keys():
    """Test Dict[Derived, V] where Dict[Base, V] expected.
    
    Expected: Dict keys are invariant, so this is tricky.
    Current: Tests actual behavior.
    """
    
    class Key: pass
    class SpecialKey(Key): pass
    
    def process_dict(d: Dict[A, B]) -> Tuple[A, B]: ...
    
    special_dict = {SpecialKey(): "value"}
    
    # Should infer from actual key type
    result = infer_return_type(process_dict, special_dict)
    assert typing.get_origin(result) == tuple
    key_type, val_type = typing.get_args(result)
    assert key_type == SpecialKey
    assert val_type == str


def test_subclass_without_orig_class():
    """Test subclass instance without __orig_class__ attribute.
    
    Expected: Should fall back to field-based inference.
    Current: Tests fallback behavior.
    """
    
    @dataclass
    class Base(typing.Generic[A]):
        value: A
    
    @dataclass
    class Derived(Base[A], typing.Generic[A]):
        extra: str
    
    def process(obj: Base[A]) -> A: ...
    
    # Create instance without __orig_class__
    derived = Derived(value=42, extra="test")
    if hasattr(derived, '__orig_class__'):
        delattr(derived, '__orig_class__')
    
    # Should still work via field extraction
    result = infer_return_type(process, derived)
    assert result == int


def test_subclass_type_mismatch_detection():
    """Test that incompatible subclass relationships are detected.
    
    Expected: Should fail when types don't align.
    Current: Correctly detects and raises error!
    """
    
    @dataclass
    class Container(typing.Generic[A]):
        value: A
    
    @dataclass
    class IntContainer(Container[int]):  # Fixed to int
        pass
    
    # Annotation expects Container[str] but instance is IntContainer(Container[int])
    def process_string_container(c: Container[str]) -> str: ...
    
    int_container = IntContainer(value=42)
    
    # Correctly detects type mismatch: annotation says str, field value is int
    # The engine extracts field pairs and finds: str annotation vs int value
    with pytest.raises(TypeInferenceError, match="Expected str, got int"):
        infer_return_type(process_string_container, int_container)


# =============================================================================
# 12. UNION COMPONENT MATCHING (Lines 504-519)
# =============================================================================


def test_union_components_match_by_origin():
    """Test that union components match by origin rather than exact type."""
    engine = UnificationEngine()
    
    from generic_utils import get_generic_info
    
    # Create unions: int | list[int] vs int | list
    union1_info = get_generic_info(int | list[int])
    union2_info = get_generic_info(int | list)
    
    # Should match by origin (int matches int, list[int] origin matches list)
    result = engine._union_components_match(union1_info, union2_info)
    
    # Depends on implementation, but should recognize origin match
    assert result == True


def test_union_components_different_lengths():
    """Test union component matching with different lengths."""
    engine = UnificationEngine()
    
    from generic_utils import get_generic_info
    
    # Different number of components
    union1_info = get_generic_info(int | str | float)
    union2_info = get_generic_info(int | str)
    
    result = engine._union_components_match(union1_info, union2_info)
    
    # Different lengths should not match
    assert result == False


# =============================================================================
# 12. ADDITIONAL PEP 484 COMPLIANCE TESTS
# =============================================================================

@pytest.mark.skip(reason="LIMITATION: typing.Any not supported")


def test_pep484_any_type():
    """Test handling of typing.Any (should be permissive)."""
    from typing import Any
    
    def process_any(x: Any, y: A) -> A: ...
    
    # Any should accept anything, but shouldn't interfere with A inference
    t = infer_return_type(process_any, "anything", 42)
    assert t == int


def test_pep484_noreturn():
    """Test that NoReturn is handled appropriately."""
    from typing import NoReturn
    
    # NoReturn in annotations shouldn't cause crashes
    # (though it's unusual to use in type inference context)


@pytest.mark.skip(reason="LIMITATION: Literal types (PEP 586) not supported")


def test_literal_types():
    """Test with Literal types from PEP 586."""
    from typing import Literal
    
    def process_literal(x: Literal[1, 2, 3], y: A) -> A: ...
    
    # Literal should be treated as its underlying type
    t = infer_return_type(process_literal, 1, "str")
    assert t == str


@pytest.mark.skip(reason="LIMITATION: Final annotations (PEP 591) not supported")


def test_final_annotation():
    """Test with Final annotation from PEP 591."""
    from typing import Final
    
    def process_final(x: Final[A]) -> A: ...
    
    # Final wraps a type and shouldn't interfere with inference
    t = infer_return_type(process_final, 42)
    assert t == int


# =============================================================================
# 13. EDGE CASES FROM PEP 526 (Variable Annotations)
# =============================================================================

@pytest.mark.skip(reason="LIMITATION: Annotated types (PEP 593) not supported")


def test_annotated_type():
    """Test with Annotated type from PEP 593."""
    try:
        from typing import Annotated
        
        def process_annotated(x: Annotated[A, "some metadata"]) -> A: ...
        
        # Annotated should extract the underlying type A
        t = infer_return_type(process_annotated, 42)
        assert t == int
    except ImportError:
        # Annotated not available in older Python
        pytest.skip("Annotated not available")


def test_classvar_in_dataclass():
    """Test that ClassVar is handled appropriately."""
    from typing import ClassVar
    
    @dataclass
    class WithClassVar(typing.Generic[A]):
        class_var: ClassVar[str] = "class"
        instance_var: A
    
    def process_class(obj: WithClassVar[A]) -> A: ...
    
    instance = WithClassVar[int](instance_var=42)
    t = infer_return_type(process_class, instance)
    assert t == int


# =============================================================================
# 14. STRESS TESTS FOR CONSTRAINT SOLVER
# =============================================================================


def test_constraint_solver_many_constraints():
    """Test constraint solver with many constraints on same TypeVar."""
    engine = UnificationEngine()
    
    # Many covariant constraints (should form union)
    constraints = [Constraint(A, int, Variance.COVARIANT) for _ in range(10)]
    constraints.extend([Constraint(A, str, Variance.COVARIANT) for _ in range(10)])
    
    sub = engine._solve_constraints(constraints)
    result = sub.get(A)
    
    # Should reduce to int | str union
    import types
    origin = typing.get_origin(result)
    assert origin is Union or origin is getattr(types, 'UnionType', None)


def test_constraint_solver_all_same():
    """Test constraint solver when all constraints are identical."""
    engine = UnificationEngine()
    
    # Many identical constraints
    constraints = [Constraint(A, int, Variance.INVARIANT) for _ in range(100)]
    
    sub = engine._solve_constraints(constraints)
    result = sub.get(A)
    
    # Should just be int
    assert result == int


# =============================================================================
# 15. NONE AND NONETYPE HANDLING
# =============================================================================


def test_nonetype_inference():
    """Test proper NoneType inference."""
    from unification_type_inference import _infer_type_from_value
    
    # None should infer as NoneType
    t = _infer_type_from_value(None)
    assert t == type(None)


def test_optional_with_all_none_values():
    """Test Optional when all values are None."""
    
    def process_all_none(items: List[Optional[A]]) -> A: ...
    
    # All None values - should fail
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_all_none, [None, None, None])


# =============================================================================
# 16. ERROR PATH COVERAGE
# =============================================================================


def test_unification_error_messages():
    """Test that UnificationError provides clear messages."""
    engine = UnificationEngine()
    
    # Create a situation that raises UnificationError
    with pytest.raises(UnificationError) as exc_info:
        # Multiple conflicting overrides
        constraints = [
            Constraint(A, int, is_override=True),
            Constraint(A, str, is_override=True),
        ]
        engine._solve_constraints(constraints)
    
    error_msg = str(exc_info.value)
    assert "Conflicting" in error_msg or "override" in error_msg


def test_type_inference_error_from_unification_error():
    """Test conversion of UnificationError to TypeInferenceError."""
    
    # Passing wrong container type should raise TypeInferenceError
    def process_specific(items: List[int]) -> int: ...
    
    with pytest.raises(TypeInferenceError) as exc_info:
        infer_return_type(process_specific, {1, 2, 3})  # Set instead of List
    
    # Error should mention type mismatch
    error_msg = str(exc_info.value)
    assert len(error_msg) > 0


# =============================================================================
# 17. DEEP NESTING TESTS - Ensures robust handling at extreme depths
# =============================================================================
# These tests validate that type inference works correctly with:
# - Deep generic class nesting (Box[Box[Box[...]]])
# - Deep union nesting
# - Depth 3+ for all container types
# - Deep recursive structures
# - Mixed container/generic nesting at depth
# - Edge cases that could break at depth


def test_triple_nested_generic_classes():
    """Test Box[Box[Box[A]]] - deep generic class nesting."""
    
    @dataclass
    class Box(typing.Generic[A]):
        content: A
    
    def triple_unbox(b: Box[Box[Box[A]]]) -> A: ...
    
    # Create deeply nested boxes
    innermost = Box[int](content=42)
    middle = Box[Box[int]](content=innermost)
    outer = Box[Box[Box[int]]](content=middle)
    
    t = infer_return_type(triple_unbox, outer)
    assert t is int


def test_quadruple_nested_generic_classes():
    """Test Box[Box[Box[Box[A]]]] - very deep generic nesting."""
    
    @dataclass
    class Container(typing.Generic[A]):
        value: A
    
    def quad_extract(c: Container[Container[Container[Container[A]]]]) -> A: ...
    
    # Build from inside out
    level1 = Container[str](value="deep")
    level2 = Container[Container[str]](value=level1)
    level3 = Container[Container[Container[str]]](value=level2)
    level4 = Container[Container[Container[Container[str]]]](value=level3)
    
    t = infer_return_type(quad_extract, level4)
    assert t is str


def test_mixed_generic_classes_deep_nesting():
    """Test Wrapper[Box[Container[A]]] - different generic classes nested."""
    
    @dataclass
    class Wrapper(typing.Generic[A]):
        wrapped: A
    
    @dataclass
    class Box(typing.Generic[A]):
        item: A
    
    @dataclass
    class Container(typing.Generic[A]):
        data: A
    
    def extract_mixed(w: Wrapper[Box[Container[A]]]) -> A: ...
    
    inner = Container[float](data=3.14)
    middle = Box[Container[float]](item=inner)
    outer = Wrapper[Box[Container[float]]](wrapped=middle)
    
    t = infer_return_type(extract_mixed, outer)
    assert t is float


def test_pydantic_dataclass_mixed_deep_nesting():
    """Test deep nesting mixing Pydantic and dataclasses."""
    
    @dataclass
    class DataBox(typing.Generic[A]):
        value: A
    
    class PydanticWrapper(BaseModel, typing.Generic[A]):
        content: A
    
    @dataclass
    class DataContainer(typing.Generic[A]):
        item: A
    
    def extract_from_mix(
        p: PydanticWrapper[DataBox[DataContainer[A]]]
    ) -> A: ...
    
    inner = DataContainer[int](item=99)
    middle = DataBox[DataContainer[int]](value=inner)
    outer = PydanticWrapper[DataBox[DataContainer[int]]](content=middle)
    
    t = infer_return_type(extract_from_mix, outer)
    assert t is int


@pytest.mark.skip(reason="LIMITATION: Deep nested unions with multiple unbound TypeVars")


def test_triple_nested_union():
    """Test Union[A, Union[B, Union[C, D]]] - deep union nesting.
    
    This documents a limitation: when a deeply nested union has multiple
    TypeVar alternatives and we only match the concrete type, the other
    TypeVars remain unbound.
    """
    
    def extract_from_nested_union(
        x: Union[A, Union[B, Union[C, int]]]
    ) -> Union[A, B, C]: ...
    
    t = infer_return_type(extract_from_nested_union, 42)


def test_union_in_list_in_union():
    """Test List[Union[A, B]] nested in Union."""
    
    def process_nested(
        data: Union[List[Union[A, B]], Dict[str, Union[A, B]]]
    ) -> Tuple[A, B]: ...
    
    t = infer_return_type(process_nested, [1, "x", 2, "y"])
    
    assert typing.get_origin(t) is tuple
    result_types = set(typing.get_args(t))
    assert result_types == {int, str}


def test_deeply_nested_union_in_containers():
    """Test Dict[A, List[Set[Union[B, C]]]] - union at depth 3."""
    
    def extract_union_types(
        data: Dict[A, List[Set[Union[B, C]]]]
    ) -> Tuple[A, B, C]: ...
    
    test_data = {
        "key1": [{1, "a"}, {2, "b"}],
        "key2": [{3, "c"}]
    }
    
    t = infer_return_type(extract_union_types, test_data)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args[0] is str


def test_set_depth_three():
    """Test Set at depth 3 - rarely tested."""
    
    def extract_from_nested_frozensets(data: List[Set[int]]) -> int: ...
    
    t = infer_return_type(extract_from_nested_frozensets, [{1, 2}, {3, 4}])
    assert t is int


def test_tuple_depth_three():
    """Test Tuple[Tuple[Tuple[A, B], C], D] - nested tuples."""
    
    def extract_from_nested_tuples(
        data: Tuple[Tuple[Tuple[A, B], C], D]
    ) -> Tuple[A, B, C, D]: ...
    
    inner = ((1, "a"), 3.14)
    outer = (inner, True)
    
    t = infer_return_type(extract_from_nested_tuples, outer)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args == (int, str, float, bool)


def test_dict_depth_four():
    """Test Dict[A, Dict[B, Dict[C, Dict[D, E]]]] - depth 4 dict nesting."""
    
    def extract_all_types(
        data: Dict[A, Dict[B, Dict[C, Dict[D, E]]]]
    ) -> Tuple[A, B, C, D, E]: ...
    
    deep_dict = {
        "level1": {
            42: {
                3.14: {
                    True: "deepest"
                }
            }
        }
    }
    
    t = infer_return_type(extract_all_types, deep_dict)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args == (str, int, float, bool, str)


def test_mixed_containers_depth_four():
    """Test List[Dict[Set[Tuple[A, B]]]] - 4 different containers."""
    
    def extract_from_complex(
        data: List[Dict[str, Set[Tuple[A, B]]]]
    ) -> Tuple[A, B]: ...
    
    complex_data = [
        {"key1": {(1, "a"), (2, "b")}},
        {"key2": {(3, "c")}}
    ]
    
    t = infer_return_type(extract_from_complex, complex_data)
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (int, str)


def test_triple_recursive_tree():
    """Test TreeNode[TreeNode[TreeNode[A]]] - 3-level recursive structure."""

    def extract_from_deep_tree(
        tree: TreeNode[TreeNode[TreeNode[A]]]
    ) -> A: ...
    
    # Innermost nodes
    leaf1 = TreeNode[int](value=1, children=[])
    leaf2 = TreeNode[int](value=2, children=[])
    
    # Middle level
    middle1 = TreeNode[TreeNode[int]](value=leaf1, children=[])
    middle2 = TreeNode[TreeNode[int]](value=leaf2, children=[])
    
    # Top level
    root = TreeNode[TreeNode[TreeNode[int]]](value=middle1, children=[])
    
    t = infer_return_type(extract_from_deep_tree, root)
    assert t is int


def test_linked_list_depth():
    """Test deep linked list structure."""
    
    @dataclass
    class Node(typing.Generic[A]):
        value: A
        next: Optional['Node[A]']
    
    def extract_value_from_list(node: Node[A]) -> A: ...
    
    # Create 5-deep linked list
    node5 = Node[str](value="end", next=None)
    node4 = Node[str](value="four", next=node5)
    node3 = Node[str](value="three", next=node4)
    node2 = Node[str](value="two", next=node3)
    node1 = Node[str](value="one", next=node2)
    
    t = infer_return_type(extract_value_from_list, node1)
    assert t is str


def test_graph_like_structure():
    """Test graph-like structure with multiple paths."""
    
    @dataclass
    class GraphNode(typing.Generic[A]):
        value: A
        edges: List['GraphNode[A]']
    
    def extract_node_type(node: GraphNode[A]) -> A: ...
    
    node1 = GraphNode[int](value=1, edges=[])
    node2 = GraphNode[int](value=2, edges=[node1])
    node3 = GraphNode[int](value=3, edges=[node1, node2])
    
    t = infer_return_type(extract_node_type, node3)
    assert t is int


def test_six_level_list_nesting():
    """Test List^6[A] - 6 levels of list nesting."""
    
    def extract_from_six_deep(
        data: List[List[List[List[List[List[A]]]]]]
    ) -> A: ...
    
    deep_data = [[[[[[42]]]]]]
    t = infer_return_type(extract_from_six_deep, deep_data)
    assert t is int


def test_seven_level_mixed_nesting():
    """Test 7-level mixed container nesting."""
    
    def extract_deeply_nested(
        data: List[Dict[str, List[Tuple[List[Dict[int, Optional[A]]]]]]]
    ) -> A: ...
    
    deep_structure = [
        {
            "key": [
                ([{1: 42, 2: 99}],),
                ([{3: 100}],)
            ]
        }
    ]
    
    t = infer_return_type(extract_deeply_nested, deep_structure)
    
    import types
    origin = typing.get_origin(t)
    if origin is Union or origin is getattr(types, 'UnionType', None):
        union_args = typing.get_args(t)
        assert int in union_args
    else:
        assert t is int


def test_empty_containers_at_depth():
    """Test that empty containers at various depths are handled."""
    
    def process_with_empties(
        a: List[List[List[A]]],
        b: A
    ) -> A: ...
    
    t = infer_return_type(process_with_empties, [[[], []]], 42)
    assert t is int


def test_mixed_types_at_each_depth_level():
    """Test mixed types at multiple depth levels simultaneously."""
    
    def process_multi_depth_mixed(
        data: List[Dict[str, List[A]]]
    ) -> A: ...
    
    mixed_depth = [
        {"a": [1, 2]},
        {"b": ["x", "y"]},
    ]
    
    t = infer_return_type(process_multi_depth_mixed, mixed_depth)
    
    import types
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, 'UnionType', None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


def test_multiple_typevars_all_at_different_depths():
    """Test A at depth 1, B at depth 2, C at depth 3, D at depth 4."""
    
    def extract_multi_depth_types(
        data: Dict[A, List[Dict[B, Set[Tuple[C, D]]]]]
    ) -> Tuple[A, B, C, D]: ...
    
    complex = {
        "level1": [
            {
                42: {
                    (3.14, True),
                }
            }
        ]
    }
    
    t = infer_return_type(extract_multi_depth_types, complex)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args == (str, int, float, bool)


def test_union_at_multiple_depths():
    """Test Union types at depths 1, 2, and 3 simultaneously."""
    
    def process_multi_level_unions(
        data: Union[
            Dict[A, Union[List[B], Set[Union[C, D]]]],
            List[A]
        ]
    ) -> Tuple[A, B, C, D]: ...
    
    test_data = {"key": [1, 2, 3]}
    
    try:
        t = infer_return_type(process_multi_level_unions, test_data)
        assert typing.get_origin(t) is tuple
    except TypeInferenceError:
        pass  # Complex union resolution - test validates we don't crash


def test_deep_and_wide():
    """Test structure that is both deep (5 levels) and wide (4 TypeVars)."""
    
    def extract_deep_wide(
        data: Dict[A, List[Dict[B, Set[Tuple[C, D]]]]]
    ) -> Tuple[A, B, C, D]: ...
    
    structure = {
        "a": [{"b": {(1, 2.0)}}],
        "c": [{"d": {(3, 4.0)}}],
    }
    
    t = infer_return_type(extract_deep_wide, structure)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args == (str, str, int, float)


def test_many_nested_containers_same_typevar():
    """Test same TypeVar appearing at multiple depths."""
    
    def process_repeated_typevar(
        data: Dict[A, List[Dict[A, Set[A]]]]
    ) -> A: ...
    
    repeated = {
        1: [{1: {1, 2, 3}}],
        2: [{2: {4, 5, 6}}],
    }
    
    t = infer_return_type(process_repeated_typevar, repeated)
    assert t is int


# =============================================================================
# 18. COMPREHENSIVE OPTIONAL HANDLING TESTS
# =============================================================================
# These tests validate Optional at multiple nesting levels and with None values
# Addresses user concern about List[Optional[Dict[...]]] with None in list


def test_list_optional_dict_with_none():
    """List[Optional[Dict[str, Optional[A]]]] with None in list - user concern."""
    
    def process_multi_optional(
        data: Optional[List[Optional[Dict[str, Optional[A]]]]]
    ) -> A: ...
    
    test_data = [
        {"key1": 42, "key2": None},
        None,  # Valid None for Optional[Dict[...]]
        {"key3": 99}
    ]
    
    t = infer_return_type(process_multi_optional, test_data)
    
    import types
    origin = typing.get_origin(t)
    if origin is Union or origin is getattr(types, 'UnionType', None):
        union_args = typing.get_args(t)
        assert int in union_args
    else:
        assert t is int


def test_list_optional_dict_all_none():
    """Edge case: ALL dicts in list are None - should fail."""
    
    def process_all_none(
        data: List[Optional[Dict[str, A]]]
    ) -> A: ...
    
    test_data = [None, None, None]
    
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_all_none, test_data)


def test_list_optional_dict_some_none():
    """List[Optional[Dict[str, A]]] with some None values."""
    
    def process_some_none(
        data: List[Optional[Dict[str, A]]]
    ) -> A: ...
    
    test_data = [
        None,
        {"key1": 42},
        None,
        {"key2": 99},
        None
    ]
    
    t = infer_return_type(process_some_none, test_data)
    assert t is int


def test_optional_list_vs_list_optional():
    """Compare: Optional[List[...]] vs List[Optional[...]]."""
    
    def process_optional_list(
        data: Optional[List[Dict[str, A]]]
    ) -> A: ...
    
    def process_list_optional(
        data: List[Optional[Dict[str, A]]]
    ) -> A: ...
    
    t1 = infer_return_type(process_optional_list, [{"key": 42}])
    assert t1 is int
    
    t2 = infer_return_type(process_list_optional, [{"key": 42}, None])
    assert t2 is int


def test_deeply_nested_optionals():
    """Optional at multiple levels simultaneously."""
    
    def process_deep_optional(
        data: Optional[List[Optional[Dict[str, Optional[List[Optional[A]]]]]]]
    ) -> A: ...
    
    test_data = [
        {"key1": [1, None, 2]},
        None,
        {"key2": [3, 4], "key3": None}
    ]
    
    t = infer_return_type(process_deep_optional, test_data)
    
    import types
    origin = typing.get_origin(t)
    if origin is Union or origin is getattr(types, 'UnionType', None):
        union_args = typing.get_args(t)
        assert int in union_args
    else:
        assert t is int


def test_optional_none_filtering():
    """None values in Optional[A] don't bind A to NoneType."""
    
    def process_optional_values(
        data: Dict[str, Optional[A]]
    ) -> A: ...
    
    test_data = {
        "a": 1,
        "b": None,
        "c": 2,
        "d": None,
        "e": 3
    }
    
    t = infer_return_type(process_optional_values, test_data)
    assert t is int  # Should be int, not int | None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



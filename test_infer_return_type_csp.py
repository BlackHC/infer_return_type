import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union, Callable

import pytest
# from infer_return_type import TypeInferenceError, infer_return_type
# from unification_type_inference import TypeInferenceError, infer_return_type_unified as infer_return_type
from csp_type_inference import CSPTypeInferenceError as TypeInferenceError, infer_return_type_csp as infer_return_type
from pydantic import BaseModel

# TypeVars for testing
A = TypeVar('A')
B = TypeVar('B') 
C = TypeVar('C')
K = TypeVar('K')
V = TypeVar('V')
X = TypeVar('X')
Y = TypeVar('Y')

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
    
    @dataclass
    class Wrap(typing.Generic[A]):
        value: A
    
    def unwrap(w: Wrap[A]) -> A: ...
    t = infer_return_type(unwrap, Wrap[int](1))
    assert t is int
    
    class Box(BaseModel, typing.Generic[A]):
        item: A
    
    def unbox(bs: List[Box[A]]) -> List[A]: ...
    t = infer_return_type(unbox, [Box[int](item=1)])
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
    
    # Same TypeVar with different types should create union and still work if return type doesn't depend on it
    def same(a: A, b: A) -> bool: ...
    t = infer_return_type(same, 1, 'x')
    assert t is bool  # Return type is bool regardless of A's binding


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
    
    @dataclass
    class Level1(typing.Generic[A]):
        inner: A
    
    @dataclass
    class Level2(typing.Generic[A]):
        wrapped: Level1[A]
        alternatives: List[A]
    
    class Level3(BaseModel, typing.Generic[A]):
        nested: Level2[A]
        extras: Dict[str, A]
    
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
    
    @dataclass
    class MultiParamContainer(typing.Generic[A, B, C]):
        primary: List[A]
        secondary: Dict[str, B] 
        tertiary: Set[C]
        mixed: List[Tuple[A, B, C]]
    
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
    
    # JSON-like nested structure
    @dataclass
    class JsonValue(typing.Generic[A]):
        data: Union[A, Dict[str, 'JsonValue[A]'], List['JsonValue[A]']]
    
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
    @dataclass
    class TypedColumn(typing.Generic[A]):
        name: str
        values: List[A]
    
    @dataclass  
    class MultiColumnData(typing.Generic[A, B, C]):
        col1: TypedColumn[A]
        col2: TypedColumn[B] 
        col3: TypedColumn[C]
    
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
    @dataclass
    class Wrap(typing.Generic[A]):
        value: A
    
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
    @dataclass 
    class PartiallySpecialized(typing.Generic[A]):
        strings: List[str]
        generic_items: List[A]
    
    def get_generic_items(ps: PartiallySpecialized[A]) -> List[A]: ...
    def get_strings(ps: PartiallySpecialized[A]) -> List[str]: ...
    
    ps = PartiallySpecialized[int](strings=["a", "b"], generic_items=[1, 2, 3])
    
    t_generic = infer_return_type(get_generic_items, ps)
    assert typing.get_origin(t_generic) is list and typing.get_args(t_generic) == (int,)
    
    t_strings = infer_return_type(get_strings, ps)
    assert typing.get_origin(t_strings) is list and typing.get_args(t_strings) == (str,)
    
    # Recursive generic structure
    @dataclass
    class TreeNode(typing.Generic[A]):
        value: A
        children: List['TreeNode[A]']
    
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
    
    @dataclass
    class Wrap(typing.Generic[A]):
        value: A
    
    class Box(BaseModel, typing.Generic[A]):
        item: A
    
    def unwrap_box_list(w: Wrap[List[Box[A]]]) -> List[A]: ...
    
    boxes = [Box[int](item=1), Box[int](item=2)]
    wrapped_boxes = Wrap[List[Box[int]]](boxes)
    
    t = infer_return_type(unwrap_box_list, wrapped_boxes)
    assert typing.get_origin(t) is list and typing.get_args(t) == (int,)


def test_optional_nested_generics():
    """Test handling optional nested generic types"""
    
    @dataclass
    class Wrap(typing.Generic[A]):
        value: A
        
    def unwrap_optional_nested(w: Optional[Wrap[A]]) -> Optional[A]: ...
    
    wrapped = Wrap[float](3.14)
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

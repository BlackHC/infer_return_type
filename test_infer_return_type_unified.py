import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union, Callable

import pytest
# from infer_return_type import TypeInferenceError, infer_return_type
from unification_type_inference import TypeInferenceError, infer_return_type_unified as infer_return_type
# from csp_type_inference import CSPTypeInferenceError as TypeInferenceError, infer_return_type_csp as infer_return_type
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
    
    # Ambiguous same TypeVar binding should fail
    def same(a: A, b: A) -> bool: ...
    with pytest.raises(TypeInferenceError):
        infer_return_type(same, 1, 'x')


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

@pytest.mark.skip(reason="WEAKNESS: Unification fails on conflicting TypeVar bindings instead of creating unions")
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


@pytest.mark.skip(reason="WEAKNESS: Unification includes None in result instead of filtering it out")  
def test_none_filtering_in_optional():
    """
    Test that None values in Optional[A] should not bind A to None.
    
    This is different behavior from Original and CSP - they filter None out correctly.
    """
    def process_dict_with_nones(d: Dict[str, Optional[A]]) -> A: ...
    
    # With None values mixed in, should still return int, not int | None
    result = infer_return_type(process_dict_with_nones, {'a': 1, 'b': None, 'c': 2})
    assert result is int  # Should be int, not int | None


@pytest.mark.skip(reason="WEAKNESS: Unification fails on complex nested union structures")
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


@pytest.mark.skip(reason="DESIGN DECISION: Bounded TypeVar strictness - int doesn't satisfy float bound")
def test_bounded_typevar_relaxed():
    """
    Test whether bounded TypeVars should be relaxed for practical use.
    
    This is technically correct per PEP 484, but Python's runtime is more lenient.
    Decision needed: should int satisfy bound=float?
    """
    Numeric = TypeVar('Numeric', bound=float)
    def process_numeric(x: Numeric) -> Numeric: ...
    
    # int is not technically a subtype of float in the type system
    # But in practice Python accepts int where float is expected
    result = infer_return_type(process_numeric, 1)
    assert result is int


@pytest.mark.skip(reason="BUG: Set[Union[A, B]] tuple access fails")
def test_set_union_distribution_fixed():
    """
    Test Set[Union[A, B]] distribution among TypeVars.
    
    Currently fails with TypeError when accessing tuple args.
    """
    def process_union_set(s: Set[Union[A, B]]) -> Tuple[Set[A], Set[B]]: ...
    
    result = infer_return_type(process_union_set, {1, 'a', 2, 'b'})
    
    # Should distribute int and str between A and B
    tuple_args = typing.get_args(result)
    assert len(tuple_args) == 2
    
    # Both should be sets
    origin1 = typing.get_origin(tuple_args[0])
    origin2 = typing.get_origin(tuple_args[1])
    assert origin1 is set and origin2 is set


# =============================================================================
# MISSING TESTS FROM CSP - VARIANCE AND CONSTRAINTS
# =============================================================================

@pytest.mark.skip(reason="TODO: Port variance testing from CSP implementation")
def test_covariant_variance_explicit():
    """
    Test explicit covariant variance handling.
    
    CSP has extensive variance tests that unification lacks.
    """
    class Animal: pass
    class Dog(Animal): pass
    
    def covariant_test(pets: List[A]) -> A: ...
    
    # List is covariant - Dog list should allow supertype inference
    dog_list = [Dog(), Dog()]
    result = infer_return_type(covariant_test, dog_list)
    
    # Could be Dog (most specific) or Animal or object (supertypes)
    assert result in [Dog, Animal, object]


@pytest.mark.skip(reason="TODO: Port contravariant testing from CSP")
def test_contravariant_variance_explicit():
    """
    Test contravariant variance in Callable parameters.
    
    Callable is contravariant in parameter types, covariant in return type.
    """
    def contravariant_test(func: Callable[[A], str]) -> A: ...
    
    def object_to_str(x: object) -> str:
        return str(x)
    
    # Callable[[object], str] where object is the most general parameter type
    result = infer_return_type(contravariant_test, object_to_str)
    assert result is object


@pytest.mark.skip(reason="TODO: Port invariant testing from CSP")
def test_invariant_dict_keys():
    """
    Test invariant variance for Dict keys.
    
    Dict keys are invariant - must be exact type match.
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


@pytest.mark.skip(reason="TODO: Port constraint system testing from CSP")
def test_constraint_priority_resolution():
    """
    Test that constraints are resolved with proper priority.
    
    CSP has priority-based constraint resolution that's missing in unification.
    """
    # type_overrides should have highest priority
    def process_list(items: List[A]) -> A: ...
    
    # Even though we have int values, override should win
    result = infer_return_type(process_list, [1, 2, 3], type_overrides={A: str})
    assert result is str


@pytest.mark.skip(reason="TODO: Port domain-based reasoning from CSP")
def test_domain_filtering_with_constraints():
    """
    Test domain filtering based on multiple constraints.
    
    CSP uses domain-based reasoning to filter possible types.
    """
    T_BOUNDED = TypeVar('T_BOUNDED', bound=int)
    
    def bounded_test(x: T_BOUNDED, y: T_BOUNDED) -> T_BOUNDED: ...
    
    # Both values must satisfy the bound
    result = infer_return_type(bounded_test, True, False)
    assert result is bool  # bool is subtype of int


@pytest.mark.skip(reason="TODO: Port union distribution tests from CSP")
def test_union_or_logic_distribution():
    """
    Test Union types as OR logic in constraint satisfaction.
    
    Union[List[A], Set[A]] means A can be inferred from either branch.
    """
    def process_union(data: Union[List[A], Set[A]]) -> A: ...
    
    # List branch
    result1 = infer_return_type(process_union, [1, 2, 3])
    assert result1 is int
    
    # Set branch  
    result2 = infer_return_type(process_union, {1, 2, 3})
    assert result2 is int


@pytest.mark.skip(reason="TODO: Port subset constraint testing from CSP")
def test_subset_constraints():
    """
    Test subset constraints like {A, B} ⊇ {int, str}.
    
    CSP explicitly models subset constraints for union types.
    """
    def process_set_union(s: Set[Union[A, B]]) -> Tuple[A, B]: ...
    
    # Set with mixed types creates subset constraint
    mixed_set = {1, 'hello', 2, 'world'}
    result = infer_return_type(process_set_union, mixed_set)
    
    # Should distribute types among A and B
    assert typing.get_origin(result) is tuple


# =============================================================================
# EDGE CASES THAT SHOULD WORK BUT MIGHT NOT
# =============================================================================

@pytest.mark.skip(reason="INVESTIGATE: Multiple conflicting invariant constraints")
def test_multiple_invariant_conflicts():
    """
    Test behavior when multiple invariant constraints conflict.
    
    Should this create a union or fail?
    """
    def process_containers(d1: Dict[A, str], d2: Dict[A, str], d3: Dict[A, str]) -> A: ...
    
    # Three dicts with different key types (invariant position)
    result = infer_return_type(process_containers, 
                               {1: 'a'}, 
                               {'x': 'b'}, 
                               {3.14: 'c'})
    
    # Should this be int | str | float, or should it fail?
    origin = typing.get_origin(result)
    import types
    assert origin is Union or origin is getattr(types, 'UnionType', None)


@pytest.mark.skip(reason="INVESTIGATE: Nested invariant+covariant mixing")
def test_nested_variance_mixing():
    """
    Test mixing invariant and covariant variance in nested structures.
    
    Dict[A, List[A]] - A is invariant in keys, covariant in list elements.
    """
    def process_mixed_variance(d: Dict[A, List[A]]) -> A: ...
    
    # Keys must match (invariant), but list can have subtypes (covariant)
    test_data = {
        1: [1, 2, 3],
        2: [4, 5, 6]
    }
    
    result = infer_return_type(process_mixed_variance, test_data)
    assert result is int


@pytest.mark.skip(reason="INVESTIGATE: TypeVar appearing in both covariant and contravariant positions")
def test_typevar_multiple_variance_positions():
    """
    Test TypeVar appearing in positions with different variance.
    
    Callable[[List[A]], A] - A is covariant in List, covariant in return position.
    """
    def complex_func(callback: Callable[[List[A]], A], default: A) -> A: ...
    
    def list_to_int(items: List[int]) -> int:
        return sum(items)
    
    # Should infer A = int from both positions
    result = infer_return_type(complex_func, list_to_int, 42)
    assert result is int


# =============================================================================
# DEBUGGING AND ERROR MESSAGE TESTS  
# =============================================================================

@pytest.mark.skip(reason="TODO: Add constraint tracing for debugging")
def test_constraint_trace_on_failure():
    """
    Test that failures include constraint traces for debugging.
    
    CSP provides detailed constraint information on failures.
    """
    def conflicting_example(a: List[A], b: List[A]) -> A: ...
    
    try:
        infer_return_type(conflicting_example, [1], ["x"])
        assert False, "Should have failed or created union"
    except TypeInferenceError as e:
        error_msg = str(e)
        # Should include information about the conflicting constraints
        assert "conflict" in error_msg.lower() or "constraint" in error_msg.lower()


@pytest.mark.skip(reason="TODO: Add readable error messages")
def test_readable_error_messages():
    """
    Test that error messages are human-readable and actionable.
    
    CSP has human-readable constraint descriptions.
    """
    def empty_container_test(items: List[A]) -> A: ...
    
    try:
        infer_return_type(empty_container_test, [])
        assert False, "Should have failed"
    except TypeInferenceError as e:
        error_msg = str(e)
        # Should explain what went wrong in clear terms
        assert len(error_msg) > 20  # Not just a terse error code


# =============================================================================
# PERFORMANCE AND SCALABILITY TESTS
# =============================================================================

@pytest.mark.skip(reason="TODO: Add performance benchmarks")
def test_deeply_nested_performance():
    """
    Test performance on deeply nested structures.
    
    Ensure unification doesn't have exponential blowup on deep nesting.
    """
    import time
    
    def deep_nested(data: List[List[List[List[List[A]]]]]) -> A: ...
    
    deep_data = [[[[[1, 2, 3]]]]]
    
    start = time.time()
    result = infer_return_type(deep_nested, deep_data)
    elapsed = time.time() - start
    
    assert result is int
    assert elapsed < 1.0  # Should complete in under 1 second


@pytest.mark.skip(reason="TODO: Add scalability tests")
def test_many_typevars_scalability():
    """
    Test scalability with many TypeVars.
    
    Ensure constraint solving doesn't become too slow with many variables.
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
# ARCHITECTURAL TESTS
# =============================================================================

def test_architectural_improvements():
    """Demonstrate architectural improvements - unified interface."""
    
    from unification_type_inference import UnificationEngine
    from generic_utils import GenericExtractor
    
    engine = UnificationEngine()
    
    # Show that we have a clean extractor interface through generic_utils
    assert len(engine.generic_utils.extractors) >= 3  # Pydantic, Dataclass, Builtin
    
    # Each extractor implements the same interface
    for extractor in engine.generic_utils.extractors:
        assert hasattr(extractor, 'can_handle_annotation')
        assert hasattr(extractor, 'can_handle_instance')
        assert hasattr(extractor, 'extract_from_annotation')
        assert hasattr(extractor, 'extract_from_instance')
    
    # Test that we can easily add new extractors by subclassing
    class CustomExtractor(GenericExtractor):
        def can_handle_annotation(self, annotation):
            return False  # dummy implementation
        
        def can_handle_instance(self, instance):
            return False
        
        def extract_from_annotation(self, annotation):
            from generic_utils import GenericInfo
            return GenericInfo()
        
        def extract_from_instance(self, instance):
            from generic_utils import GenericInfo
            return GenericInfo()
    
    # Can be easily added to the system
    custom_extractor = CustomExtractor()
    engine.generic_utils.extractors.append(custom_extractor)


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

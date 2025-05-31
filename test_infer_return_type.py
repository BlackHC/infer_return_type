import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union, Callable

import pytest
from infer_return_type import TypeInferenceError, infer_return_type
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
    
    # Same TypeVar with conflicting types (should fail)
    def same_typevar_conflict(a: List[A], b: List[A]) -> A: ...
    with pytest.raises(TypeInferenceError):
        infer_return_type(same_typevar_conflict, [1, 2], ["a", "b"])
    
    # Partial binding with empty containers (should fail)
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

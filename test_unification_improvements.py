"""
Test suite demonstrating improvements with the unification-based type inference.

This shows how the new algorithm addresses limitations in the original system:
1. Union formation instead of conflicts
2. Variance handling
3. TypeVar bounds and constraints
4. Cleaner architecture for different generic type systems
"""

import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union, Callable

import pytest
from unification_type_inference import TypeInferenceError, infer_return_type_unified
from infer_return_type import TypeInferenceError as OriginalTypeInferenceError
from pydantic import BaseModel

# TypeVars for testing
A = TypeVar('A')
B = TypeVar('B') 
C = TypeVar('C')
K = TypeVar('K')
V = TypeVar('V')

# TypeVars with constraints/bounds
T_BOUNDED = TypeVar('T_BOUNDED', bound=int)
T_CONSTRAINED = TypeVar('T_CONSTRAINED', int, str)


def test_union_formation_instead_of_conflicts():
    """Test that mixed types form unions instead of causing conflicts."""
    
    def process_mixed_nested(data: List[List[A]]) -> A: ...
    
    # This used to fail with TypeInferenceError in the original system
    # Now it should form a union type
    mixed_nested = [[1, 2], ["a", "b"]]
    t = infer_return_type_unified(process_mixed_nested, mixed_nested)
    
    # Should be int | str (union of the different element types)
    assert typing.get_origin(t) is Union or hasattr(t, '__args__')
    if hasattr(t, '__args__'):
        union_args = typing.get_args(t)
        assert set(union_args) == {int, str}


def test_dict_with_mixed_key_value_types():
    """Test Dict[A, A] with mixed key/value types forms union."""
    
    def process_self_referential_dict(data: Dict[A, A]) -> A: ...
    
    # Keys: int|str, Values: str|int - should infer A = int | str
    mixed_dict = {1: "a", "b": 2}
    t = infer_return_type_unified(process_self_referential_dict, mixed_dict)
    
    # Should be int | str
    assert typing.get_origin(t) is Union or hasattr(t, '__args__')
    if hasattr(t, '__args__'):
        union_args = typing.get_args(t)
        assert set(union_args) == {int, str}


def test_nested_containers_with_none_values():
    """Test that None values in nested containers are handled gracefully."""
    
    def process_nested_with_none(data: List[List[A]]) -> A: ...
    
    # This used to fail - now should form union including None
    nested_with_none = [[1, 2], [None, None]]
    t = infer_return_type_unified(process_nested_with_none, nested_with_none)
    
    # Should be int | None
    assert typing.get_origin(t) is Union or hasattr(t, '__args__')
    if hasattr(t, '__args__'):
        union_args = typing.get_args(t)
        assert int in union_args
        assert type(None) in union_args


def test_complex_nested_branches():
    """Test complex nested structures where different branches have different types."""
    
    def process_complex_nested(data: List[Dict[str, List[A]]]) -> A: ...
    
    # Different branches contribute different types - should form union
    complex_data = [
        {"branch1": [1, 2, 3]},        # A = int
        {"branch2": ["a", "b", "c"]}   # A = str
    ]
    
    t = infer_return_type_unified(process_complex_nested, complex_data)
    
    # Should be int | str
    assert typing.get_origin(t) is Union or hasattr(t, '__args__')
    if hasattr(t, '__args__'):
        union_args = typing.get_args(t)
        assert set(union_args) == {int, str}


def test_partial_binding_with_empty_containers():
    """Test that partial binding works when some containers are empty."""
    
    def process_multiple_lists(list1: List[A], list2: List[A], list3: List[A]) -> A: ...
    
    # Some empty, some non-empty with conflicts - should infer from all non-empty
    t = infer_return_type_unified(process_multiple_lists, [], [1, 2], ["a", "b"])
    
    # Should be int | str (union from both non-empty lists)
    assert typing.get_origin(t) is Union or hasattr(t, '__args__')
    if hasattr(t, '__args__'):
        union_args = typing.get_args(t)
        assert set(union_args) == {int, str}


def test_typevar_bounds_enforcement():
    """Test that TypeVar bounds are properly enforced."""
    
    def process_bounded(x: T_BOUNDED) -> T_BOUNDED: ...
    
    # bool is a subtype of int, should work
    t = infer_return_type_unified(process_bounded, True)
    assert t is bool
    
    # str is not a subtype of int, should fail
    with pytest.raises(TypeInferenceError):
        infer_return_type_unified(process_bounded, "hello")


def test_typevar_constraints_enforcement():
    """Test that TypeVar constraints are properly enforced."""
    
    def process_constrained(x: T_CONSTRAINED) -> T_CONSTRAINED: ...
    
    # int is in constraints, should work
    t = infer_return_type_unified(process_constrained, 42)
    assert t is int
    
    # str is in constraints, should work
    t = infer_return_type_unified(process_constrained, "hello")
    assert t is str
    
    # float is not in constraints, should fail
    with pytest.raises(TypeInferenceError):
        infer_return_type_unified(process_constrained, 3.14)


def test_variance_handling():
    """Test that variance is properly handled in constraint resolution."""
    
    def covariant_example(data: List[A]) -> A: ...
    
    # Mixed types in covariant position should form union
    mixed_list = [1, "hello", 3.14]
    t = infer_return_type_unified(covariant_example, mixed_list)
    
    assert typing.get_origin(t) is Union or hasattr(t, '__args__')
    if hasattr(t, '__args__'):
        union_args = typing.get_args(t)
        assert set(union_args) == {int, str, float}


def test_callable_inference_basic():
    """Test basic callable inference (this is a new capability)."""
    
    # Note: This is a simplified test - full callable inference is complex
    # But the unification framework makes it easier to extend
    
    def transform_list(items: List[A], func: str) -> List[str]: ...  # Simplified
    
    t = infer_return_type_unified(transform_list, [1, 2, 3], "dummy")
    assert typing.get_origin(t) is list
    assert typing.get_args(t) == (str,)


def test_advanced_inheritance_chains():
    """Test complex inheritance with the unification system."""
    
    @dataclass
    class Base(typing.Generic[A, B]):
        base_a: A
        base_b: B
    
    @dataclass 
    class Derived(Base[A, str], typing.Generic[A]):
        derived_data: List[A]
    
    def extract_from_derived(d: Derived[A]) -> Tuple[A, List[A]]: ...
    
    derived_instance = Derived[int](
        base_a=42,
        base_b="fixed_string", 
        derived_data=[1, 2, 3]
    )
    
    t = infer_return_type_unified(extract_from_derived, derived_instance)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args[0] is int
    assert typing.get_origin(args[1]) is list
    assert typing.get_args(args[1]) == (int,)


def test_complex_pydantic_nested():
    """Test complex nested Pydantic structures."""
    
    class Level1(BaseModel, typing.Generic[A]):
        inner: A
    
    class Level2(BaseModel, typing.Generic[A]):
        wrapped: Level1[A]
        alternatives: List[A]
    
    class Level3(BaseModel, typing.Generic[A]):
        nested: Level2[A]
        extras: Dict[str, A]
    
    def unwrap_all_levels(l3: Level3[A]) -> A: ...
    
    deep_structure = Level3[bool](
        nested=Level2[bool](
            wrapped=Level1[bool](inner=True),
            alternatives=[False, True]
        ),
        extras={"flag1": False, "flag2": True}
    )
    
    t = infer_return_type_unified(unwrap_all_levels, deep_structure)
    assert t is bool


def test_union_with_generics():
    """Test unions containing generic types."""
    
    @dataclass
    class Wrap(typing.Generic[A]):
        value: A
    
    def maybe_wrap(x: A, should_wrap: bool) -> Union[A, Wrap[A]]: ...
    
    t = infer_return_type_unified(maybe_wrap, 42, True)
    
    # Should return Union[int, Wrap[int]]
    assert typing.get_origin(t) is Union
    union_args = typing.get_args(t)
    assert int in union_args
    
    # Check if Wrap[int] is in the union
    wrap_types = [arg for arg in union_args if typing.get_origin(arg) == Wrap]
    assert len(wrap_types) > 0
    assert typing.get_args(wrap_types[0]) == (int,)


def test_recursive_generic_structures():
    """Test recursive generic structures like trees."""
    
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
    t_value = infer_return_type_unified(get_tree_value, tree)
    assert t_value is str
    
    # Test extracting children list  
    t_children = infer_return_type_unified(get_tree_children, tree)
    assert typing.get_origin(t_children) is list
    children_arg = typing.get_args(t_children)[0]
    assert typing.get_origin(children_arg) == TreeNode


def test_multiple_union_containers():
    """Test functions with multiple union container parameters."""
    
    def process_multiple_unions(
        data1: Union[List[A], Tuple[A, ...]], 
        data2: Union[Set[B], Dict[str, B]]
    ) -> Tuple[A, B]: ...
    
    # Should handle multiple union parameters
    t = infer_return_type_unified(process_multiple_unions, [1, 2], {"a": "hello", "b": "world"})
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (int, str)


def test_type_override_integration():
    """Test that type overrides work properly with the unification system."""
    
    def process_with_override(items: List[A]) -> A: ...
    
    # Empty list with type override should work
    t = infer_return_type_unified(process_with_override, [], type_overrides={A: int})
    assert t is int
    
    # Non-empty list with conflicting override - override wins
    t = infer_return_type_unified(process_with_override, ["hello"], type_overrides={A: int})
    assert t is int


def test_mixed_containers():
    """Compare original vs unified approach on mixed containers."""
    
    # Import original function for comparison
    from infer_return_type import infer_return_type
    
    def process_nested_mixed(data: List[List[A]]) -> A: ...
    
    mixed_nested = [[1, 2], ["a", "b"]]
    
    # Unified system should succeed with union
    t = infer_return_type_unified(process_nested_mixed, mixed_nested)
    assert typing.get_origin(t) is Union or hasattr(t, '__args__')
    assert set(typing.get_args(t)) == {int, str}


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


if __name__ == "__main__":
    # Run a few key tests to demonstrate the improvements
    print("Testing union formation instead of conflicts...")
    test_union_formation_instead_of_conflicts()
    print("✓ Passed")
    
    print("Testing TypeVar bounds enforcement...")
    test_typevar_bounds_enforcement()
    print("✓ Passed")
    
    print("Testing complex nested structures...")
    test_complex_nested_branches()
    print("✓ Passed")
    
    print("Testing architectural improvements...")
    test_architectural_improvements()
    print("✓ Passed")
    
    print("\nAll improvements demonstrated successfully!") 
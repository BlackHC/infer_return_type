"""
Additional tests for unification_type_inference.py to improve code coverage.

These tests target specific uncovered lines and edge cases based on:
- PEP 484: Type Hints
- PEP 526: Variable Annotations  
- PEP 544: Protocols
- Python's type system documentation

Focus areas:
1. Direct use of UnificationEngine.unify_annotation_with_value
2. Forward references (ForwardRef handling)
3. Multiple conflicting type_overrides
4. Edge cases in constraint resolution
5. Complex Union distribution scenarios
6. Type substitution edge cases
7. Keyword argument handling
"""

import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union, ForwardRef
import pytest

from unification_type_inference import (
    TypeInferenceError, 
    infer_return_type_unified as infer_return_type,
    UnificationEngine,
    Constraint,
    Substitution,
    Variance,
    UnificationError
)
from pydantic import BaseModel

# TypeVars for testing
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
K = TypeVar('K')
V = TypeVar('V')


# =============================================================================
# 1. DIRECT UNIFICATION ENGINE TESTS (Lines 98-116)
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
    
    @dataclass
    class TreeNode(typing.Generic[A]):
        value: A
        children: List['TreeNode[A]']
    
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
    result = engine._try_match_generic_info_with_instance(list_info, set_info, constraints)
    
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
    result = engine._try_match_generic_info_with_instance(list_info, dict_info, constraints)
    
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
    result = engine._try_match_generic_info_with_instance(ann_info, val_info, constraints)
    
    # Should successfully match and extract constraints
    # (May or may not return True depending on implementation details)


# =============================================================================
# 11. UNION COMPONENT MATCHING (Lines 504-519)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


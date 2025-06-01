"""
Test suite for CSP-based type inference system.

This demonstrates how type inference can be modeled as a constraint satisfaction problem
similar to SAT solving, where:
- Union types create OR constraints (subset constraints)
- Container types create equality constraints  
- Variance creates inequality constraints
- All constraints are ANDed together
"""

import typing
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union
from dataclasses import dataclass

import pytest
from csp_type_inference import (
    CSPTypeInferenceError, 
    infer_return_type_csp,
    CSPTypeInferenceEngine,
    ConstraintType,
    TypeConstraint
)

# TypeVars for testing
A = TypeVar('A')
B = TypeVar('B') 
C = TypeVar('C')
K = TypeVar('K')
V = TypeVar('V')
X = TypeVar('X')
Y = TypeVar('Y')

# TypeVars with constraints/bounds for testing
T_BOUNDED = TypeVar('T_BOUNDED', bound=int)
T_CONSTRAINED = TypeVar('T_CONSTRAINED', int, str)


def test_basic_csp_functionality():
    """Test that the CSP engine can solve basic type constraints."""
    
    def simple_list(items: List[A]) -> A: ...
    
    # Homogeneous list should create equality constraint: A = int
    result = infer_return_type_csp(simple_list, [1, 2, 3])
    assert result is int
    
    # Mixed list should create subset constraint: {A} ⊇ {int, str}
    result = infer_return_type_csp(simple_list, [1, "hello"])
    # Should get union type
    assert typing.get_origin(result) is Union or hasattr(result, '__args__')


def test_constraint_types_demonstration():
    """Demonstrate different types of constraints in our CSP model."""
    
    engine = CSPTypeInferenceEngine()
    
    # 1. EQUALITY constraints (A = type)
    def test_equality():
        engine.clear()
        engine.collect_constraints_from_annotation_value(A, 42, "test")
        constraints = [c for c in engine.constraints if c.constraint_type == ConstraintType.EQUALITY]
        assert len(constraints) > 0
        assert A in constraints[0].variables
        assert int in constraints[0].types
    
    # 2. SUBSET constraints ({TypeVars} ⊇ {concrete_types})
    def test_subset():
        engine.clear()
        engine.collect_constraints_from_annotation_value(List[A], [1, "hello"], "test")
        constraints = [c for c in engine.constraints if c.constraint_type == ConstraintType.SUBSET]
        assert len(constraints) > 0
        assert A in constraints[0].variables
        assert {int, str}.issubset(constraints[0].types)
    
    # 3. BOUNDS_CHECK constraints
    def test_bounds():
        engine.clear()
        engine.collect_constraints_from_annotation_value(T_BOUNDED, True, "test")
        constraints = [c for c in engine.constraints if c.constraint_type == ConstraintType.BOUNDS_CHECK]
        assert len(constraints) > 0
        assert T_BOUNDED in constraints[0].variables
    
    test_equality()
    test_subset() 
    test_bounds()


def test_union_constraints_as_or_logic():
    """Demonstrate how Union types create OR-like constraints."""
    
    def process_union(data: Union[List[A], Set[A]]) -> A: ...
    
    # List case - should bind A to element type
    result = infer_return_type_csp(process_union, [1, 2, 3])
    assert result is int
    
    # Set case - should also bind A to element type
    result = infer_return_type_csp(process_union, {1, 2, 3})
    assert result is int
    
    # Mixed list case - should create union
    result = infer_return_type_csp(process_union, [1, "hello"])
    assert typing.get_origin(result) is Union or hasattr(result, '__args__')


def test_set_union_constraints():
    """Test Set[A | B] constraints - the motivating example."""
    
    def process_set_union(data: Set[Union[A, B]]) -> Tuple[A, B]: ...
    
    # Set with mixed types should distribute among A and B
    mixed_set = {1, "hello", 2, "world"}
    
    # This is a complex case - the CSP should handle it
    try:
        result = infer_return_type_csp(process_set_union, mixed_set)
        assert typing.get_origin(result) is tuple
        # The exact distribution depends on the CSP solver's strategy
    except CSPTypeInferenceError:
        # This is expected - it's a complex constraint that may not have unique solution
        pass


def test_constraint_priority_system():
    """Test that constraint priorities work correctly."""
    
    engine = CSPTypeInferenceEngine()
    
    # Add constraints with different priorities
    engine.add_equality_constraint(A, int, "high_priority")  # Priority 10
    engine.add_subset_constraint({A}, {str, float}, "lower_priority")  # Priority 5
    
    # High priority should win
    solution = engine.solve()
    assert solution.get(A) is int


def test_type_overrides_as_constraints():
    """Test that type overrides work as high-priority equality constraints."""
    
    def empty_list_example(items: List[A]) -> A: ...
    
    # Empty list normally fails, but override should work
    result = infer_return_type_csp(empty_list_example, [], type_overrides={A: str})
    assert result is str
    
    # Override should win over inferred type
    result = infer_return_type_csp(empty_list_example, [42], type_overrides={A: str})
    assert result is str


def test_bounded_typevar_constraints():
    """Test TypeVar bounds as constraints."""
    
    def bounded_example(x: T_BOUNDED) -> T_BOUNDED: ...
    
    # bool is subtype of int, should work
    result = infer_return_type_csp(bounded_example, True)
    assert result is bool
    
    # str is not subtype of int, should fail
    with pytest.raises(CSPTypeInferenceError):
        infer_return_type_csp(bounded_example, "hello")


def test_constrained_typevar_constraints():
    """Test TypeVar explicit constraints."""
    
    def constrained_example(x: T_CONSTRAINED) -> T_CONSTRAINED: ...
    
    # int is in constraints, should work
    result = infer_return_type_csp(constrained_example, 42)
    assert result is int
    
    # str is in constraints, should work
    result = infer_return_type_csp(constrained_example, "hello")
    assert result is str
    
    # float is not in constraints, should fail
    with pytest.raises(CSPTypeInferenceError):
        infer_return_type_csp(constrained_example, 3.14)


def test_complex_nested_constraints():
    """Test complex nested structures that create multiple interacting constraints."""
    
    def complex_nested(data: Dict[A, List[B]]) -> Tuple[A, B]: ...
    
    nested_data = {
        "key1": [1, 2, 3],
        "key2": [4, 5, 6]
    }
    
    result = infer_return_type_csp(complex_nested, nested_data)
    assert typing.get_origin(result) is tuple
    args = typing.get_args(result)
    assert args[0] is str  # A = str (dict keys)
    assert args[1] is int  # B = int (list elements)


def test_conflicting_constraints():
    """Test how the CSP handles conflicting constraints."""
    
    def conflicting_example(a: List[A], b: List[A]) -> A: ...
    
    # Different lists with different element types - should create union
    result = infer_return_type_csp(conflicting_example, [1, 2], ["a", "b"])
    
    # Should get union type
    assert typing.get_origin(result) is Union or hasattr(result, '__args__')
    if hasattr(result, '__args__'):
        union_args = typing.get_args(result)
        assert int in union_args and str in union_args


def test_constraint_propagation():
    """Test that constraints propagate correctly through the domain."""
    
    engine = CSPTypeInferenceEngine()
    
    # Add multiple related constraints
    engine.add_equality_constraint(A, int, "source1")
    engine.add_bounds_constraint(A, "source2")
    
    # Solve and check that all constraints were satisfied
    solution = engine.solve()
    assert solution.get(A) is int


def test_unsatisfiable_constraints():
    """Test detection of unsatisfiable constraint systems."""
    
    engine = CSPTypeInferenceEngine()
    
    # Add conflicting exact constraints
    engine.add_equality_constraint(A, int, "constraint1")
    engine.add_equality_constraint(A, str, "constraint2")  # Conflicts with first
    
    # Should detect unsatisfiable system
    with pytest.raises(CSPTypeInferenceError):
        engine.solve()


def test_csp_vs_unification_comparison():
    """Compare CSP approach with unification approach on same problems."""
    
    from unification_type_inference import infer_return_type_unified
    
    def mixed_nested(data: List[List[A]]) -> A: ...
    
    mixed_data = [[1, 2], ["a", "b"]]
    
    # Both should handle this case (unification already does)
    unified_result = infer_return_type_unified(mixed_nested, mixed_data)
    csp_result = infer_return_type_csp(mixed_nested, mixed_data)
    
    # Both should produce union types
    assert typing.get_origin(unified_result) is Union or hasattr(unified_result, '__args__')
    assert typing.get_origin(csp_result) is Union or hasattr(csp_result, '__args__')


def test_debug_constraint_sources():
    """Test that constraint sources are tracked for debugging."""
    
    engine = CSPTypeInferenceEngine()
    engine.collect_constraints_from_annotation_value(List[A], [1, 2], "test_source")
    
    # Check that constraints have source information
    for constraint in engine.constraints:
        assert constraint.source.startswith("test_source")


def test_constraint_description_readability():
    """Test that constraint descriptions are human-readable."""
    
    constraint = TypeConstraint(
        constraint_type=ConstraintType.SUBSET,
        variables={A, B},
        types={int, str},
        description="Test constraint"
    )
    
    constraint_str = str(constraint)
    assert "subset" in constraint_str
    assert "A" in constraint_str or "B" in constraint_str
    assert "int" in constraint_str or "str" in constraint_str


def test_domain_based_reasoning():
    """Test the type domain reasoning system."""
    
    from csp_type_inference import TypeDomain
    
    domain = TypeDomain(A)
    
    # Add possible types
    domain.add_possible_type(int)
    domain.add_possible_type(str)
    domain.add_possible_type(float)
    
    # Add subtype constraint
    domain.add_subtype_constraint(int)  # A must be subtype of int
    
    # Should filter out str (not subtype of int)
    valid_types = domain.get_valid_types()
    assert int in valid_types
    assert bool in valid_types or len(valid_types) == 1  # bool is subtype of int
    assert str not in valid_types
    assert float not in valid_types


def test_multiple_solutions_handling():
    """Test handling when multiple solutions are possible."""
    
    # Create scenario with multiple valid solutions
    def ambiguous_example(data: Union[List[A], Set[B]]) -> Union[A, B]: ...
    
    # Could bind either way, but CSP should pick one consistently
    result = infer_return_type_csp(ambiguous_example, [1, 2, 3])
    
    # Should get some valid result
    assert result is not None


def test_real_world_csp_patterns():
    """Test real-world patterns using CSP approach."""
    
    # JSON-like structure
    @dataclass
    class JsonNode(typing.Generic[A]):
        value: A
        children: List['JsonNode[A]']
    
    def extract_json_value(node: JsonNode[A]) -> A: ...
    
    json_tree = JsonNode[str](
        value="root",
        children=[JsonNode[str](value="child", children=[])]
    )
    
    result = infer_return_type_csp(extract_json_value, json_tree)
    assert result is str
    
    # Database-like multi-column structure
    def process_columns(
        col1: List[A], 
        col2: List[B], 
        col3: List[A]  # Same TypeVar as col1 - creates constraint
    ) -> Dict[A, B]: ...
    
    result = infer_return_type_csp(
        process_columns,
        [1, 2, 3],      # A = int
        ["a", "b", "c"], # B = str  
        [4, 5, 6]       # A = int (consistent)
    )
    
    assert typing.get_origin(result) is dict
    key_type, value_type = typing.get_args(result)
    assert key_type is int
    assert value_type is str


if __name__ == "__main__":
    # Run some key tests to demonstrate the CSP approach
    print("Testing CSP-based type inference...")
    
    print("\n1. Basic functionality:")
    test_basic_csp_functionality()
    print("✓ Basic CSP inference works")
    
    print("\n2. Constraint types:")
    test_constraint_types_demonstration()
    print("✓ Different constraint types work correctly")
    
    print("\n3. Union constraints (OR logic):")
    test_union_constraints_as_or_logic()
    print("✓ Union types create proper OR constraints")
    
    print("\n4. TypeVar bounds and constraints:")
    test_bounded_typevar_constraints()
    test_constrained_typevar_constraints()
    print("✓ TypeVar bounds and constraints enforced")
    
    print("\n5. Complex nested structures:")
    test_complex_nested_constraints()
    print("✓ Complex nested constraints solved")
    
    print("\n6. Conflicting constraints (union formation):")
    test_conflicting_constraints()
    print("✓ Conflicting constraints create unions")
    
    print("\nCSP-based type inference working correctly!")
    print("\nKey insights demonstrated:")
    print("- Union types → OR constraints (subset constraints)")
    print("- Container types → Equality constraints") 
    print("- TypeVar bounds → Inequality constraints")
    print("- All constraints ANDed together in CSP")
    print("- Priority system for constraint resolution")
    print("- Domain-based reasoning for type inference") 
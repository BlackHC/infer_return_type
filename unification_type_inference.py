"""
Unification-based type inference system for generic function return types.

This implements a formal unification algorithm that can handle:
1. Complex nested generic structures
2. TypeVar bounds and constraints  
3. Variance (covariance/contravariance)
4. Union formation when conflicts arise
5. Common interface for different generic type systems

The key insight is to treat type inference as a constraint satisfaction problem
where we unify annotation structures with concrete value types.
"""

import inspect
import types
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union
from enum import Enum
from collections import defaultdict

# Import unified generic utilities
from generic_utils import (
    get_generic_info, get_instance_generic_info, create_union_if_needed, get_annotation_value_pairs
)


class UnificationError(Exception):
    """Raised when unification fails."""


class TypeInferenceError(Exception):
    """Raised when type inference fails."""


class Variance(Enum):
    """Type variance for generic parameters."""
    COVARIANT = "covariant"
    CONTRAVARIANT = "contravariant"
    INVARIANT = "invariant"


class Constraint:
    """Represents a type constraint between a TypeVar and a concrete type."""
    
    def __init__(self, typevar: TypeVar, concrete_type: type, variance: Variance = Variance.INVARIANT, is_override: bool = False):
        self.typevar = typevar
        self.concrete_type = concrete_type
        self.variance = variance
        self.is_override = is_override
    
    def __str__(self):
        override_str = " (override)" if self.is_override else ""
        return f"{self.typevar} ~ {self.concrete_type} ({self.variance.value}){override_str}"
    
    def __repr__(self):
        return self.__str__()


class Substitution:
    """Represents a substitution of TypeVars to concrete types."""
    
    def __init__(self):
        self.bindings: Dict[TypeVar, type] = {}
    
    def bind(self, typevar: TypeVar, concrete_type: type):
        """Bind a TypeVar to a concrete type."""
        self.bindings[typevar] = concrete_type
    
    def get(self, typevar: TypeVar) -> Optional[type]:
        """Get the binding for a TypeVar."""
        return self.bindings.get(typevar)
    
    def apply(self, annotation: Any) -> Any:
        """Apply this substitution to an annotation."""
        return _substitute_typevars(annotation, self.bindings)
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose this substitution with another."""
        result = Substitution()
        # Apply other to our bindings first
        for tv, typ in self.bindings.items():
            result.bind(tv, other.apply(typ) if isinstance(typ, TypeVar) else typ)
        # Add other's bindings
        for tv, typ in other.bindings.items():
            if tv not in result.bindings:
                result.bind(tv, typ)
        return result
    
    def __str__(self):
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.bindings.items()) + "}"


class UnificationEngine:
    """Core unification engine for type inference."""
    
    def __init__(self):
        pass
    
    def unify_annotation_with_value(
        self, 
        annotation: Any, 
        value: Any,
        constraints: List[Constraint] = None
    ) -> Substitution:
        """
        Unify an annotation with a concrete value to produce TypeVar bindings.
        
        This is the main entry point for type inference.
        """
        if constraints is None:
            constraints = []
        
        # Collect constraints from the annotation/value pair
        self._collect_constraints(annotation, value, constraints)
        
        # Solve the constraint system
        return self._solve_constraints(constraints)
    
    def _collect_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Recursively collect type constraints from annotation/value pairs."""
        
        try:
            self._collect_constraints_internal(annotation, value, constraints)
        except UnificationError as e:
            # Convert to TypeInferenceError for consistency
            raise TypeInferenceError(str(e))
    
    def _collect_constraints_internal(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Internal constraint collection that can raise UnificationError."""
        
        # Base case: Direct TypeVar
        if isinstance(annotation, TypeVar):
            concrete_type = _infer_type_from_value(value)
            constraints.append(Constraint(annotation, concrete_type))
            return
        
        info = get_generic_info(annotation)
        
        # Special case: Union types (need custom logic for alternative selection)
        if info.origin is Union or (hasattr(types, 'UnionType') and info.origin is getattr(types, 'UnionType')):
            self._handle_union_constraints(annotation, value, constraints)
            return
        
        # Try unified extraction first for standard containers and custom types
        pairs = get_annotation_value_pairs(annotation, value)
        if pairs:
            self._collect_constraints_from_pairs(pairs, annotation, value, constraints)
            return
        
        # Fallback: Try direct generic structure matching for custom types
        # This handles cases like Pydantic[A, list[B]] with Pydantic[int, list[str]]
        if self._try_direct_structure_match(annotation, value, constraints):
            return
        
        # Final fallback: handle non-container types or types without type parameters
        if info.origin and info.origin is not type(value):
            # Special case: ForwardRef that couldn't be resolved
            # Try to match by name with value's type
            if hasattr(info.origin, '__forward_arg__'):
                # It's a ForwardRef - check if value's class name matches
                value_class_name = type(value).__name__
                forward_name = info.origin.__forward_arg__
                # Extract class name from ForwardRef (e.g., 'TreeNode[A]' → 'TreeNode')
                ref_class_name = forward_name.split('[')[0] if '[' in forward_name else forward_name
                if value_class_name == ref_class_name:
                    # Names match - use direct structure matching with instance
                    if self._try_direct_structure_match(annotation, value, constraints):
                        return
                    # If structure matching didn't work, just skip (assume match by name)
                    return
            
            # Provide more specific error messages for common container mismatches
            origin_name = info.origin.__name__ if hasattr(info.origin, '__name__') else str(info.origin)
            value_type_name = type(value).__name__
            raise UnificationError(f"Expected {origin_name}, got {value_type_name}")
    
    def _add_covariant_constraints_for_elements(
        self, typevar: TypeVar, values, constraints: List[Constraint]
    ):
        """
        Add separate covariant constraints for each distinct type in values.
        
        This allows proper union formation and bound checking in the constraint solver.
        Uses _infer_type_from_value to get full generic types, not just base types.
        """
        element_types = {_infer_type_from_value(item) for item in values}
        for element_type in element_types:
            constraints.append(Constraint(typevar, element_type, Variance.COVARIANT))
    
    def _collect_constraints_from_pairs(
        self, 
        pairs: List[Tuple[Any, Any]], 
        annotation: Any,  # noqa: ARG002 - kept for API consistency
        value: Any,  # noqa: ARG002 - kept for API consistency
        constraints: List[Constraint]
    ):
        """Process (GenericInfo, value) pairs into constraints.
        
        This handles both simple TypeVar mappings and complex nested structures.
        Special handling for Union types within containers (List[Union[A, B]], Set[Union[A, B]], etc.)
        """
        # Group pairs by TypeVar for better constraint generation
        typevar_values = defaultdict(list)
        complex_pairs = []
        
        for generic_info, val in pairs:
            if isinstance(generic_info.origin, TypeVar):
                typevar_values[generic_info.origin].append(val)
            else:
                complex_pairs.append((generic_info, val))
        
        # Create covariant constraints for TypeVars (allows union formation)
        for typevar, values in typevar_values.items():
            if values:
                self._add_covariant_constraints_for_elements(typevar, values, constraints)
        
        # Handle complex cases (non-TypeVar pairs)
        if complex_pairs:
            # Check if we have multiple pairs with the same Union type for distribution
            # (e.g., List[Union[A, B]] with multiple elements)
            if len(complex_pairs) > 1:
                first_info, _ = complex_pairs[0]
                first_resolved = get_generic_info(first_info.resolved_type)
                
                # If this is a Union type shared by multiple values, try distribution
                if first_resolved.origin is Union or (hasattr(types, 'UnionType') and first_resolved.origin is getattr(types, 'UnionType')):
                    # Check if all pairs have the same Union structure
                    all_same_union = all(
                        get_generic_info(gi.resolved_type).origin in (Union, getattr(types, 'UnionType', None))
                        for gi, _ in complex_pairs
                    )
                    
                    if all_same_union:
                        # Collect all values for distribution
                        all_values = [val for _, val in complex_pairs]
                        union_args = first_resolved.concrete_args
                        
                        # Try to distribute types among TypeVars in the union
                        if self._try_distribute_union_types(all_values, union_args, constraints):
                            return
                        
                        # Fallback: match each value to union alternatives
                        for _, val in complex_pairs:
                            self._match_value_to_union_alternatives(val, union_args, constraints)
                        return
            
            # For single pairs or non-uniform unions, process each individually
            for generic_info, val in complex_pairs:
                # Use GenericInfo-based matching instead of resolved_type
                # to avoid losing TypeVars due to Pydantic same-TypeVar optimization
                if self._try_match_generic_info_with_instance(generic_info, val, constraints):
                    continue
                # Fallback: recursively collect constraints
                self._collect_constraints_internal(generic_info.resolved_type, val, constraints)
    
    def _try_match_generic_info_with_instance(
        self, 
        annotation_info: Any,  # GenericInfo
        value: Any, 
        constraints: List[Constraint]
    ) -> bool:
        """Match GenericInfo directly with instance to extract TypeVar bindings.
        
        This avoids the Pydantic same-TypeVar optimization issue where Box[A].resolved_type
        returns Box instead of Box[A], losing the TypeVar information.
        """
        # Get instance type information
        val_info = get_instance_generic_info(value)
        
        # Both must be generic with compatible origins
        if not (annotation_info.is_generic and val_info.is_generic):
            return False
        
        if annotation_info.origin != val_info.origin:
            return False
        
        if len(annotation_info.concrete_args) != len(val_info.concrete_args):
            return False
        
        # Match each type argument pair
        found_constraints = False
        for ann_arg, val_arg in zip(annotation_info.concrete_args, val_info.concrete_args):
            # Direct TypeVar binding
            if isinstance(ann_arg.origin, TypeVar) and not isinstance(val_arg.origin, TypeVar):
                constraints.append(Constraint(ann_arg.origin, val_arg.resolved_type, Variance.INVARIANT))
                found_constraints = True
            # Recursive matching for nested structures
            elif ann_arg.is_generic and val_arg.is_generic:
                if self._try_match_generic_info_with_instance(ann_arg, val_arg, constraints):
                    found_constraints = True
        
        return found_constraints
    
    def _try_direct_structure_match(self, annotation: Any, value: Any, constraints: List[Constraint]) -> bool:
        """Try to match generic structures directly (e.g., Generic[A, B] with Generic[int, str]).
        
        This is a fallback for cases where get_annotation_value_pairs doesn't give us useful constraints.
        """
        ann_info = get_generic_info(annotation)
        val_info = get_instance_generic_info(value)
        
        # Both must be generic with compatible origins and matching argument counts
        if not (ann_info.is_generic and val_info.is_generic):
            return False
            
        if ann_info.origin != val_info.origin:
            return False
        
        if len(ann_info.concrete_args) != len(val_info.concrete_args):
            return False
        
        # Match each type argument pair
        found_constraints = False
        for ann_arg, val_arg in zip(ann_info.concrete_args, val_info.concrete_args):
            if isinstance(ann_arg.origin, TypeVar) and not isinstance(val_arg.origin, TypeVar):
                constraints.append(Constraint(ann_arg.origin, val_arg.resolved_type, Variance.INVARIANT))
                found_constraints = True
            elif ann_arg.origin == val_arg.origin and ann_arg.concrete_args and val_arg.concrete_args:
                # Recursively match nested structures
                if self._try_direct_structure_match(ann_arg.resolved_type, val_arg.resolved_type, constraints):
                    found_constraints = True
        
        return found_constraints
    
    def _handle_union_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Union type constraints by trying each alternative."""
        args = get_generic_info(annotation).concrete_args
        
        # Try each union alternative
        best_constraints = None
        best_score = -1
        
        for alternative_info in args:
            try:
                temp_constraints = []
                if isinstance(alternative_info.origin, TypeVar):
                    # Direct TypeVar alternative - use inferred type from value
                    concrete_type = _infer_type_from_value(value)
                    temp_constraints.append(Constraint(alternative_info.origin, concrete_type, Variance.INVARIANT))
                else:
                    # Resolved type alternative
                    alternative = alternative_info.resolved_type
                    self._collect_constraints_internal(alternative, value, temp_constraints)
                
                # Score this alternative - prefer structured matches over direct TypeVar matches
                # Structured matches (like List[A], Dict[K,V]) provide more specific constraints
                score = len(temp_constraints)
                
                # Bonus points for matching structured types (not just bare TypeVar)
                if not isinstance(alternative_info.origin, TypeVar):
                    # Check if the alternative structure matches the value structure
                    alt_origin = get_generic_info(alternative_info.resolved_type).origin
                    value_type = type(value)
                    if alt_origin and alt_origin == value_type:
                        # Perfect structure match - prefer this
                        score += 100
                
                if score > best_score:
                    best_score = score
                    best_constraints = temp_constraints
            except (UnificationError, TypeError):
                continue
        
        if best_constraints is not None:
            constraints.extend(best_constraints)
        else:
            raise UnificationError(f"Value {value} doesn't match any alternative in {annotation}")
    
    def _solve_constraints(self, constraints: List[Constraint]) -> Substitution:
        """Solve the constraint system to produce a substitution with global context awareness."""
        
        substitution = Substitution()
        
        # Group constraints by TypeVar
        constraint_groups = defaultdict(list)
        for constraint in constraints:
            constraint_groups[constraint.typevar].append(constraint)
        
        # First pass: resolve TypeVars with unambiguous constraints
        resolved_in_first_pass = set()
        
        for typevar, typevar_constraints in constraint_groups.items():
            if self._can_resolve_unambiguously(typevar_constraints):
                resolved_type = self._resolve_typevar_constraints(typevar, typevar_constraints)
                substitution.bind(typevar, resolved_type)
                resolved_in_first_pass.add(typevar)
        
        # Second pass: resolve remaining TypeVars using context from first pass
        for typevar, typevar_constraints in constraint_groups.items():
            if typevar not in resolved_in_first_pass:
                # Try to use context from already resolved TypeVars
                refined_constraints = self._refine_constraints_with_context(
                    typevar_constraints, substitution
                )
                resolved_type = self._resolve_typevar_constraints(typevar, refined_constraints)
                substitution.bind(typevar, resolved_type)
        
        return substitution
    
    def _can_resolve_unambiguously(self, constraints: List[Constraint]) -> bool:
        """Check if constraints can be resolved without ambiguity."""
        if len(constraints) <= 1:
            return True
        
        # If all constraints have the same concrete type, unambiguous
        concrete_types = [c.concrete_type for c in constraints]
        if len(set(concrete_types)) == 1:
            return True
        
        # If we have only invariant constraints with different types, this is ambiguous (conflict)
        variances = [c.variance for c in constraints]
        if all(v == Variance.INVARIANT for v in variances):
            return len(set(concrete_types)) == 1  # Only unambiguous if all same type
        
        # If we have overrides, those are unambiguous
        if any(c.is_override for c in constraints):
            return True
        
        # Covariant constraints can be resolved (by union formation)
        return True
    
    def _refine_constraints_with_context(
        self, 
        constraints: List[Constraint], 
        context: Substitution  # noqa: ARG002 - reserved for future optimizations
    ) -> List[Constraint]:
        """Refine constraints using context from already resolved TypeVars.
        
        Currently a no-op, but provides an extension point for future optimizations
        such as constraint propagation based on already-resolved TypeVars.
        """
        return constraints
    
    def _resolve_typevar_constraints(self, typevar: TypeVar, constraints: List[Constraint]) -> type:
        """Resolve constraints for a single TypeVar."""
        
        if len(constraints) == 1:
            constraint = constraints[0]
            return self._check_typevar_bounds(typevar, constraint.concrete_type)
        
        # Check if we have any override constraints
        override_constraints = [c for c in constraints if c.is_override]
        
        # If we have override constraints, they take precedence
        if override_constraints:
            if len(override_constraints) == 1:
                # Single override - use it
                return self._check_typevar_bounds(typevar, override_constraints[0].concrete_type)
            else:
                # Multiple overrides - they must be consistent
                override_types = [c.concrete_type for c in override_constraints]
                if len(set(override_types)) == 1:
                    return self._check_typevar_bounds(typevar, override_types[0])
                else:
                    raise UnificationError(f"Conflicting override constraints for {typevar}: {override_constraints}")
        
        # No overrides - handle normally
        concrete_types = [c.concrete_type for c in constraints]
        
        # Check if all constraints are the same
        if len(set(concrete_types)) == 1:
            return self._check_typevar_bounds(typevar, concrete_types[0])
        
        # Different constraints - distinguish between "forced unions" and "conflicting sources"
        # - Forced unions: single container with mixed types (List[A] with mixed elements)
        # - Conflicting sources: multiple separate containers claiming different types for same TypeVar
        
        # Check if constraints come from the same "source context" or different sources
        covariant_constraints = [c for c in constraints if c.variance == Variance.COVARIANT]
        invariant_constraints = [c for c in constraints if c.variance == Variance.INVARIANT]
        
        # If we have covariant constraints (like List[A] with mixed elements), form union
        # Preserve type precision - only collapse to supertype if TypeVar bound requires it
        if covariant_constraints and not invariant_constraints:
            union_type = create_union_if_needed(set(concrete_types))
            return self._check_typevar_bounds(typevar, union_type)
        
        # If we have multiple invariant constraints with different types, form a union
        # This handles cases like: def identity(a: A, b: A) -> A with identity(1, 'x')
        # Result should be int | str, not an error
        if len(invariant_constraints) > 1:
            invariant_types = [c.concrete_type for c in invariant_constraints]
            if len(set(invariant_types)) > 1:
                # Multiple independent sources with different types - create union
                return self._check_typevar_bounds(typevar, create_union_if_needed(set(invariant_types)))
        
        # Mixed variance - default to union formation
        return self._check_typevar_bounds(typevar, create_union_if_needed(set(concrete_types)))
    
    def _matches_any_constraint(self, inferred_type: type, constraints: tuple) -> bool:
        """Check if inferred type matches any of the constraints.
        
        Handles:
        - Exact match: int == int
        - Origin match: list[int] matches list
        - Union match: (list[int] | float) matches (float | list) by comparing origins
        """
        inferred_info = get_generic_info(inferred_type)
        
        # Check each constraint
        for constraint in constraints:
            # Exact match
            if inferred_type == constraint:
                return True
            
            # Origin match for generic types (list[int] matches list)
            constraint_info = get_generic_info(constraint)
            if inferred_info.origin == constraint_info.origin:
                # If constraint is bare type (list) and inferred is specialized (list[int]), accept it
                if not constraint_info.is_generic and inferred_info.is_generic:
                    return True
                # If both have same structure, check recursively (handled by exact match above)
                elif inferred_info.is_generic and constraint_info.is_generic:
                    # Both generic - check if they're equivalent
                    if inferred_type == constraint:
                        return True
            
            # Union constraint matching: check if inferred union components match constraint union components
            if (inferred_info.origin is Union or (hasattr(types, 'UnionType') and inferred_info.origin is getattr(types, 'UnionType'))) and \
               (constraint_info.origin is Union or (hasattr(types, 'UnionType') and constraint_info.origin is getattr(types, 'UnionType'))):
                # Both are unions - check if components match by origin
                if self._union_components_match(inferred_info, constraint_info):
                    return True
        
        return False
    
    def _union_components_match(self, inferred_union_info, constraint_union_info) -> bool:
        """Check if union components match by comparing origins.
        
        Accepts list[int] as matching list, etc.
        """
        inferred_components = inferred_union_info.concrete_args
        constraint_components = constraint_union_info.concrete_args
        
        if len(inferred_components) != len(constraint_components):
            return False
        
        # Extract origins from both sides (list[int] → list)
        inferred_origins = {comp.origin for comp in inferred_components}
        constraint_origins = {comp.origin for comp in constraint_components}
        
        return inferred_origins == constraint_origins
    
    def _check_typevar_bounds(self, typevar: TypeVar, concrete_type: type) -> type:
        """Check if concrete type satisfies TypeVar bounds and constraints.
        
        Per PEP 484, constrained TypeVars must resolve to exactly ONE of the specified types,
        not a union of them. Union types are rejected for constrained TypeVars.
        """
        
        # Get type information using generic_utils
        type_info = get_generic_info(concrete_type)
        origin = type_info.origin
        
        # Check explicit constraints (e.g., TypeVar('T', int, str))
        if typevar.__constraints__:
            # For union types, check if it matches any constraint (which may also be unions)
            if origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType')):
                # Check if the union matches any of the constraint unions
                if not self._matches_any_constraint(concrete_type, typevar.__constraints__):
                    raise UnificationError(
                        f"Type {concrete_type} violates constraints {typevar.__constraints__} for {typevar}"
                    )
            # For non-union types, check direct match or origin match
            elif not self._matches_any_constraint(concrete_type, typevar.__constraints__):
                raise UnificationError(
                    f"Type {concrete_type} violates constraints {typevar.__constraints__} for {typevar}"
                )
        
        # Check bound (e.g., TypeVar('T', bound=int))
        if typevar.__bound__:
            # For union types, check if all components satisfy the bound
            if origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType')):
                union_args = type_info.concrete_args
                # All components must satisfy the bound
                for arg_info in union_args:
                    arg_type = arg_info.resolved_type
                    if not _is_subtype(arg_type, typevar.__bound__):
                        raise UnificationError(
                            f"Type {arg_type} in union {concrete_type} doesn't satisfy bound {typevar.__bound__} for {typevar}"
                        )
            else:
                # Single type must satisfy the bound
                if not _is_subtype(concrete_type, typevar.__bound__):
                    raise UnificationError(
                        f"Type {concrete_type} doesn't satisfy bound {typevar.__bound__} for {typevar}"
                    )
        
        return concrete_type

    def _get_existing_typevar_bindings(self, constraints: List[Constraint], variance_filter: Variance = Variance.INVARIANT) -> Dict[TypeVar, Set[type]]:
        """Extract existing TypeVar bindings from constraints for context-aware resolution.
        
        Args:
            constraints: List of constraints to analyze
            variance_filter: Only consider constraints with this variance (default: INVARIANT for strong evidence)
            
        Returns:
            Dictionary mapping TypeVars to sets of types they're constrained to
        """
        bindings = defaultdict(set)
        for constraint in constraints:
            if constraint.variance == variance_filter:
                bindings[constraint.typevar].add(constraint.concrete_type)
        return dict(bindings)
    
    def _try_distribute_union_types(self, values, union_alternatives: List, constraints: List[Constraint]) -> bool:
        """
        Try to distribute types from values among TypeVars in a Union.
        
        For Set[Union[A, B]] with values {1, 'a', 2, 'b'}, distribute types so that
        A=int and B=str (or vice versa), rather than A=int|str and B=int|str.
        
        Args:
            values: Iterable of values (set, list, etc.)
            
        Returns True if distribution was successful, False otherwise.
        """
        # Only works if all union alternatives are TypeVars
        typevars = [alt.origin for alt in union_alternatives if isinstance(alt.origin, TypeVar)]
        if len(typevars) != len(union_alternatives):
            return False  # Some alternatives are not TypeVars
        
        # Collect distinct types from values
        value_types = {type(v) for v in values}
        
        # Simple heuristic: if number of types equals number of TypeVars, distribute them
        if len(value_types) == len(typevars):
            # Sort for deterministic assignment
            sorted_types = sorted(value_types, key=lambda t: t.__name__)
            sorted_typevars = sorted(typevars, key=lambda tv: tv.__name__)
            
            # Assign one type to each TypeVar with INVARIANT variance
            # INVARIANT ensures that each TypeVar gets exactly one type
            for typevar, concrete_type in zip(sorted_typevars, sorted_types):
                constraints.append(Constraint(typevar, concrete_type, Variance.INVARIANT))
            
            return True
        
        # Can't distribute evenly, fall back to default behavior
        return False
    
    def _match_value_to_union_alternatives(self, value: Any, union_alternatives: List, constraints: List[Constraint]):
        """Match a value against union alternatives and collect constraints."""
        value_type = type(value)
        
        # First, check if the value exactly matches any concrete (non-TypeVar) type in the union
        # This handles cases like Optional[A] where None should match the concrete None type
        for alt_info in union_alternatives:
            if not isinstance(alt_info.origin, TypeVar) and alt_info.resolved_type == value_type:
                # Perfect match with concrete type - no constraints needed
                return
        
        # Get existing TypeVar bindings for context-aware matching
        existing_bindings = self._get_existing_typevar_bindings(constraints, Variance.INVARIANT)
        
        # Try to assign this value to the TypeVar that already has evidence for this type
        matched_typevar = None
        for alt_info in union_alternatives:
            if isinstance(alt_info.origin, TypeVar) and alt_info.origin in existing_bindings:
                existing_types = existing_bindings[alt_info.origin]
                if len(existing_types) == 1 and value_type in existing_types:
                    # Perfect match - this TypeVar already has evidence for this exact type
                    matched_typevar = alt_info.origin
                    break
        
        if matched_typevar:
            # Add a covariant constraint since this is coming from a Set/collection
            constraints.append(Constraint(matched_typevar, value_type, Variance.COVARIANT))
            return
        
        # No perfect match - check if we can rule out some TypeVars based on conflicting evidence
        ruled_out = set()
        for alt_info in union_alternatives:
            if isinstance(alt_info.origin, TypeVar) and alt_info.origin in existing_bindings:
                existing_types = existing_bindings[alt_info.origin]
                if len(existing_types) == 1 and value_type not in existing_types:
                    # This TypeVar has strong evidence for a different type
                    ruled_out.add(alt_info.origin)
        
        # Add constraints for remaining candidates
        candidates = [alt_info.origin for alt_info in union_alternatives 
                     if isinstance(alt_info.origin, TypeVar) and alt_info.origin not in ruled_out]
        
        if candidates:
            # Use covariant constraints to allow union formation if needed
            for candidate in candidates:
                constraints.append(Constraint(candidate, value_type, Variance.COVARIANT))
        else:
            # Fallback: add constraints for all TypeVar alternatives with invariant variance
            for alt_info in union_alternatives:
                if isinstance(alt_info.origin, TypeVar):
                    constraints.append(Constraint(alt_info.origin, value_type, Variance.INVARIANT))


def _is_subtype(subtype: type, supertype: type) -> bool:
    """Check if subtype is a subtype of supertype."""
    try:
        return issubclass(subtype, supertype)
    except TypeError:
        # Handle cases where subtype might not be a class
        return False


def _infer_type_from_value(value: Any) -> type:
    """Infer the most specific type from a value."""
    if value is None:
        return type(None)
    
    base_type = type(value)
    
    # For collections, try to infer element types
    if isinstance(value, list) and value:
        element_types = {type(item) for item in value}
        if len(element_types) == 1:
            element_type = list(element_types)[0]
            return list[element_type]
        else:
            return list[create_union_if_needed(element_types)]
    elif isinstance(value, dict) and value:
        key_types = {type(k) for k in value.keys()}
        value_types = {type(v) for v in value.values()}
        
        key_type = list(key_types)[0] if len(key_types) == 1 else create_union_if_needed(key_types)
        value_type = list(value_types)[0] if len(value_types) == 1 else create_union_if_needed(value_types)
        
        return dict[key_type, value_type]
    elif isinstance(value, tuple):
        element_types = tuple(type(item) for item in value)
        return tuple[element_types]
    elif isinstance(value, set) and value:
        element_types = {type(item) for item in value}
        if len(element_types) == 1:
            element_type = list(element_types)[0]
            return set[element_type]
        else:
            return set[create_union_if_needed(element_types)]
    
    return base_type


def _substitute_typevars(annotation: Any, bindings: Dict[TypeVar, type]) -> Any:
    """Substitute TypeVars in an annotation with their bindings."""
    
    if isinstance(annotation, TypeVar):
        if annotation in bindings:
            bound_value = bindings[annotation]
            # If bound value is GenericInfo, extract its resolved_type
            if hasattr(bound_value, 'resolved_type'):
                return bound_value.resolved_type
            return bound_value
        else:
            # Instead of failing, return the TypeVar as-is - this will be caught later
            return annotation
    
    # Use generic_utils to get type information
    type_info = get_generic_info(annotation)
    origin = type_info.origin
    args_info = type_info.concrete_args
    
    if not origin or not args_info:
        return annotation
    
    # Extract resolved types from GenericInfo objects
    args = [arg_info.resolved_type for arg_info in args_info]
    
    # Handle Union types specially - only include bound TypeVars
    if origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType')):
        substituted_args = []
        
        for arg in args:
            substituted_arg = _substitute_typevars(arg, bindings)
            # Only include the arg if it doesn't contain unbound TypeVars
            if not _has_unbound_typevars(substituted_arg):
                substituted_args.append(substituted_arg)
        
        # If we have at least one bound arg, return the union of bound args
        if substituted_args:
            if len(substituted_args) == 1:
                return substituted_args[0]
            # Use helper function to create Union type
            return create_union_if_needed(set(substituted_args))
        
        # If no args were bound, return the original annotation (will be caught as unbound)
        return annotation
    
    # Recursively substitute in type arguments
    substituted_args = []
    for arg in args:
        substituted_args.append(_substitute_typevars(arg, bindings))
    
    # Reconstruct the type using modern syntax
    if origin in (list, List):
        return list[substituted_args[0]]
    elif origin in (dict, Dict):
        return dict[substituted_args[0], substituted_args[1]]
    elif origin in (tuple, Tuple):
        return tuple[tuple(substituted_args)]
    elif origin in (set, Set):
        return set[substituted_args[0]]
    elif origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType')):
        if len(substituted_args) == 1:
            return substituted_args[0]
        return create_union_if_needed(set(substituted_args))
    else:
        # For other generic types, try to reconstruct
        try:
            return origin[tuple(substituted_args)]
        except Exception:
            return annotation


def infer_return_type_unified(
    fn: callable,
    *args,
    type_overrides: Optional[Dict[TypeVar, type]] = None,
    **kwargs,
) -> type:
    """
    Infer the concrete return type using unification algorithm.
    
    This is the main entry point that replaces the original infer_return_type.
    """
    
    if type_overrides is None:
        type_overrides = {}
    
    # Get function signature and return annotation
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation
    
    if return_annotation is inspect.Signature.empty:
        raise ValueError("Function must have return type annotation")
    
    # Create unification engine
    engine = UnificationEngine()
    
    # Collect all constraints from function parameters
    all_constraints = []
    
    # Process positional arguments
    param_names = list(sig.parameters.keys())
    for i, arg in enumerate(args):
        if i < len(param_names):
            param = sig.parameters[param_names[i]]
            if param.annotation != inspect.Parameter.empty:
                engine._collect_constraints(param.annotation, arg, all_constraints)
    
    # Process keyword arguments
    for name, value in kwargs.items():
        if name in sig.parameters:
            param = sig.parameters[name]
            if param.annotation != inspect.Parameter.empty:
                engine._collect_constraints(param.annotation, value, all_constraints)
    
    # Add type overrides as constraints
    for typevar, override_type in type_overrides.items():
        all_constraints.append(Constraint(typevar, override_type, is_override=True))
    
    # Solve constraints to get substitution
    try:
        substitution = engine._solve_constraints(all_constraints)
    except UnificationError as e:
        raise TypeInferenceError(str(e))
    
    # Apply substitution to return annotation
    result = substitution.apply(return_annotation)
    
    # Handle case where TypeVars remain unbound (e.g., empty containers)
    if isinstance(result, TypeVar):
        raise TypeInferenceError(f"Could not infer type for {result} - insufficient type information")
    
    # Check for any remaining unbound TypeVars in complex types
    if _has_unbound_typevars(result):
        raise TypeInferenceError(f"Could not fully infer return type - some TypeVars remain unbound: {result}")
    
    return result


def _has_unbound_typevars(annotation: Any) -> bool:
    """Check if an annotation contains any unbound TypeVars."""
    if isinstance(annotation, TypeVar):
        return True
    
    # Use generic_utils for consistent handling
    type_info = get_generic_info(annotation)
    
    if type_info.concrete_args:
        # Recursively check each type argument
        for arg_info in type_info.concrete_args:
            if _has_unbound_typevars(arg_info.resolved_type):
                return True
        return False
    
    return False 
"""
CSP-based type inference system that explicitly models type inference as constraint satisfaction.

This implementation treats type inference as a SAT-like problem where:
1. Union types create OR constraints: Set[A | B | str] with {1, 1.0, "hello"} means {A, B, str} ⊇ {int, float, str}
2. Container types create equality constraints: List[A] with [1, 2, 3] means A = int  
3. Variance creates inequality constraints: covariant allows A ≤ SuperType, contravariant allows A ≥ SubType
4. All constraints are ANDed together (must be satisfied simultaneously)
5. When multiple solutions exist, we prefer minimal/most specific ones

The key insight: Type unification is essentially solving a constraint satisfaction problem
in the domain of types rather than boolean variables.
"""

import inspect
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum

# Import unified generic utilities
from generic_utils import (
    GenericTypeUtils, get_concrete_args, get_generic_origin, create_union_if_needed
)


class CSPTypeInferenceError(Exception):
    """Raised when CSP-based type inference fails."""


class ConstraintType(Enum):
    """Types of constraints in our CSP model."""
    EQUALITY = "equality"           # A = int (exact assignment)
    SUBSET = "subset"              # {A, B} ⊇ {int, str} (union constraint)  
    SUBTYPE = "subtype"            # A ≤ SuperType (covariant constraint)
    SUPERTYPE = "supertype"        # A ≥ SubType (contravariant constraint)
    BOUNDS_CHECK = "bounds"        # A satisfies TypeVar bounds
    EXCLUSION = "exclusion"        # A ≠ type (negative constraint)


class Variance(Enum):
    """Type variance for generic parameters."""
    COVARIANT = "covariant"         # A ≤ SuperType (List[T] is covariant in T)
    CONTRAVARIANT = "contravariant" # A ≥ SubType (Callable[[T], R] is contravariant in T) 
    INVARIANT = "invariant"         # A = ExactType (Dict[K, V] is invariant in K)


# Variance rules for common generic types
VARIANCE_MAP = {
    list: [Variance.COVARIANT],                    # List[T] - covariant in T
    List: [Variance.COVARIANT],
    dict: [Variance.INVARIANT, Variance.COVARIANT], # Dict[K, V] - invariant in K, covariant in V
    Dict: [Variance.INVARIANT, Variance.COVARIANT],
    tuple: [Variance.COVARIANT],                   # Tuple[T, ...] - covariant in T
    Tuple: [Variance.COVARIANT],                   # Note: fixed tuples are more complex
    set: [Variance.COVARIANT],                     # Set[T] - covariant in T
    Set: [Variance.COVARIANT],
    Callable: [Variance.CONTRAVARIANT, Variance.COVARIANT], # Callable[[T], R] - contravariant in T, covariant in R
}


@dataclass
class TypeConstraint:
    """A single constraint in our CSP model."""
    constraint_type: ConstraintType
    variables: Set[TypeVar]  # TypeVars involved in this constraint
    types: Set[type]        # Concrete types involved
    description: str        # Human-readable description
    priority: int = 1       # Higher priority constraints are satisfied first
    source: str = ""        # Where this constraint came from (for debugging)
    variance: Variance = Variance.INVARIANT  # Variance context for this constraint
    
    def __str__(self):
        vars_str = ", ".join(str(v) for v in self.variables)
        types_str = ", ".join(str(t) for t in self.types)
        variance_str = self.variance.value if hasattr(self.variance, 'value') else str(self.variance)
        return f"{self.constraint_type.value}: vars={{{vars_str}}} types={{{types_str}}} - {self.description} ({variance_str})"


@dataclass  
class CSPSolution:
    """A solution to the CSP type inference problem."""
    bindings: Dict[TypeVar, type] = field(default_factory=dict)
    confidence: float = 1.0  # How confident we are in this solution (0-1)
    conflicts: List[str] = field(default_factory=list)  # Any conflicts resolved
    
    def bind(self, typevar: TypeVar, concrete_type: type):
        """Bind a TypeVar to a concrete type."""
        self.bindings[typevar] = concrete_type
    
    def get(self, typevar: TypeVar) -> Optional[type]:
        """Get binding for a TypeVar."""
        return self.bindings.get(typevar)
    
    def apply(self, annotation: Any) -> Any:
        """Apply this solution to substitute TypeVars in an annotation."""
        return _substitute_typevars(annotation, self.bindings)


class TypeDomain:
    """Represents the domain of possible types for a TypeVar in our CSP."""
    
    def __init__(self, typevar: TypeVar):
        self.typevar = typevar
        self.possible_types: Set[type] = set()
        self.excluded_types: Set[type] = set()
        self.must_be_subtype_of: Set[type] = set()
        self.must_be_supertype_of: Set[type] = set()
        self.exact_type: Optional[type] = None
        
    def add_possible_type(self, t: type):
        """Add a type to the possible domain."""
        self.possible_types.add(t)
        
    def exclude_type(self, t: type):
        """Exclude a type from the domain."""
        self.excluded_types.add(t)
        
    def set_exact_type(self, t: type):
        """Set the exact type (direct assignment - no automatic union creation)."""
        self.exact_type = t
        
    def add_subtype_constraint(self, supertype: type):
        """Add constraint that this TypeVar must be subtype of given type."""
        self.must_be_subtype_of.add(supertype)
        
    def add_supertype_constraint(self, subtype: type):
        """Add constraint that this TypeVar must be supertype of given type."""
        self.must_be_supertype_of.add(subtype)
        
    def get_valid_types(self) -> Set[type]:
        """Get all currently valid types for this TypeVar."""
        if self.exact_type is not None:
            # Check if exact type satisfies all constraints
            candidate = {self.exact_type}
            
            # Apply subtype constraints
            if self.must_be_subtype_of:
                candidate = {t for t in candidate if any(_is_subtype(t, super_t) for super_t in self.must_be_subtype_of)}
                
            # Apply supertype constraints  
            if self.must_be_supertype_of:
                candidate = {t for t in candidate if any(_is_subtype(sub_t, t) for sub_t in self.must_be_supertype_of)}
                
            return candidate
            
        valid = self.possible_types.copy()
        
        # Remove excluded types
        valid -= self.excluded_types
        
        # Apply subtype constraints
        if self.must_be_subtype_of:
            valid = {t for t in valid if any(_is_subtype(t, super_t) for super_t in self.must_be_subtype_of)}
            
        # Apply supertype constraints  
        if self.must_be_supertype_of:
            valid = {t for t in valid if any(_is_subtype(sub_t, t) for sub_t in self.must_be_supertype_of)}
            
        return valid
        
    def is_empty(self) -> bool:
        """Check if domain is empty (unsatisfiable)."""
        return len(self.get_valid_types()) == 0
        
    def is_singleton(self) -> bool:
        """Check if domain has exactly one valid type."""
        return len(self.get_valid_types()) == 1
        
    def get_best_type(self) -> type:
        """Get the best type from the domain (most specific if multiple options)."""
        valid = self.get_valid_types()
        if not valid:
            raise CSPTypeInferenceError(f"No valid types for {self.typevar}")
        if len(valid) == 1:
            return next(iter(valid))
        
        # Multiple valid types - prefer the most specific one for practical inference
        # For supertype constraints (covariant), prefer the observed type over its supertypes
        if self.must_be_supertype_of:
            # Find the most specific type that satisfies all supertype constraints
            observed_types = self.must_be_supertype_of
            for observed_type in observed_types:
                if observed_type in valid:
                    # The observed type itself is valid and most specific
                    return observed_type
        
        # For subtype constraints (contravariant), prefer the observed type as well
        if self.must_be_subtype_of:
            # Try to find a specific type rather than the most general supertype
            observed_supertypes = self.must_be_subtype_of
            # Look for the most specific type that is still a subtype
            for candidate in sorted(valid, key=lambda t: len(t.__mro__) if hasattr(t, '__mro__') else 0, reverse=True):
                if all(_is_subtype(candidate, supertype) for supertype in observed_supertypes):
                    return candidate
        
        # Fallback: if no clear most specific type, create union
        # But first try to find a single type that makes sense
        
        # Remove object if there are more specific types
        if object in valid and len(valid) > 1:
            specific_types = valid - {object}
            if len(specific_types) == 1:
                return next(iter(specific_types))
        
        # Create union of remaining types
        return create_union_if_needed(valid)


class CSPTypeInferenceEngine:
    """Main CSP solver for type inference."""
    
    def __init__(self):
        self.constraints: List[TypeConstraint] = []
        self.domains: Dict[TypeVar, TypeDomain] = {}
        self.solutions: List[CSPSolution] = []
        # Track constraint priorities for each domain  
        self.domain_priorities: Dict[TypeVar, int] = {}
        # Track constraint sources for each domain
        self.domain_sources: Dict[TypeVar, str] = {}
        # Use unified generic type utilities
        self.generic_utils = GenericTypeUtils()
        
    def clear(self):
        """Clear all constraints and domains for fresh solving."""
        self.constraints.clear()
        self.domains.clear()
        self.solutions.clear()
        self.domain_priorities.clear()
        self.domain_sources.clear()
        
    def add_constraint(self, constraint: TypeConstraint):
        """Add a constraint to the CSP."""
        self.constraints.append(constraint)
        
        # Initialize domains for any new TypeVars
        for var in constraint.variables:
            if var not in self.domains:
                self.domains[var] = TypeDomain(var)
                
    def add_equality_constraint(self, typevar: TypeVar, concrete_type: type, source: str = "", variance: Variance = Variance.INVARIANT):
        """Add A = type constraint."""
        # Type overrides get highest priority
        priority = 15 if source == "override" else 10
        constraint = TypeConstraint(
            constraint_type=ConstraintType.EQUALITY,
            variables={typevar},
            types={concrete_type},
            description=f"{typevar} = {concrete_type}",
            priority=priority,  # High priority - exact constraints
            source=source,
            variance=variance
        )
        self.add_constraint(constraint)
        
    def add_subset_constraint(self, typevars: Set[TypeVar], concrete_types: Set[type], variance: Variance = Variance.COVARIANT, source: str = ""):
        """Add {TypeVars} ⊇ {concrete_types} constraint (union constraint)."""
        constraint = TypeConstraint(
            constraint_type=ConstraintType.SUBSET,
            variables=typevars,
            types=concrete_types,
            description=f"{{{', '.join(str(v) for v in typevars)}}} ⊇ {{{', '.join(str(t) for t in concrete_types)}}}",
            priority=5,
            source=source,
            variance=variance
        )
        self.add_constraint(constraint)
        
    def add_subtype_constraint(self, typevar: TypeVar, supertype: type, source: str = ""):
        """Add A ≤ SuperType constraint (covariant)."""
        constraint = TypeConstraint(
            constraint_type=ConstraintType.SUBTYPE,
            variables={typevar},
            types={supertype},
            description=f"{typevar} ≤ {supertype}",
            priority=7,
            source=source,
            variance=Variance.COVARIANT
        )
        self.add_constraint(constraint)
        
    def add_supertype_constraint(self, typevar: TypeVar, subtype: type, source: str = ""):
        """Add A ≥ SubType constraint (contravariant)."""
        constraint = TypeConstraint(
            constraint_type=ConstraintType.SUPERTYPE,
            variables={typevar},
            types={subtype},
            description=f"{typevar} ≥ {subtype}",
            priority=7,
            source=source,
            variance=Variance.CONTRAVARIANT
        )
        self.add_constraint(constraint)
        
    def add_bounds_constraint(self, typevar: TypeVar, source: str = ""):
        """Add TypeVar bounds checking constraint.""" 
        constraint = TypeConstraint(
            constraint_type=ConstraintType.BOUNDS_CHECK,
            variables={typevar},
            types=set(),
            description=f"Check bounds for {typevar}",
            priority=8,
            source=source
        )
        self.add_constraint(constraint)
        
    def collect_constraints_from_annotation_value(self, annotation: Any, value: Any, source: str = ""):
        """Collect constraints from annotation/value pair (main entry point)."""
        self._collect_constraints_recursive(annotation, value, source)
        
    def _collect_constraints_recursive(self, annotation: Any, value: Any, source: str):
        """Recursively collect constraints from annotation/value pairs."""
        
        # Base case: Direct TypeVar
        if isinstance(annotation, TypeVar):
            concrete_type = _infer_type_from_value(value)
            self.add_equality_constraint(annotation, concrete_type, f"{source}:direct", Variance.INVARIANT)
            self.add_bounds_constraint(annotation, f"{source}:bounds")
            return
            
        # Use generic_utils for consistent type information extraction
        origin = get_generic_origin(annotation)
        args_info = get_concrete_args(annotation)
        
        # Extract raw types from GenericInfo objects for backward compatibility
        args = [arg_info.resolved_type for arg_info in args_info]
        
        # Handle Union types - these create subset constraints
        if origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType')):
            self._handle_union_annotation(annotation, value, source)
            return
            
        # Handle Optional (Union[T, None])
        if origin is Union and len(args) == 2 and type(None) in args:
            if value is not None:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                self._collect_constraints_recursive(non_none_type, value, f"{source}:optional")
            # If value is None, we can't infer much but don't fail
            return
            
        # Handle all generic containers using unified approach
        if origin and args_info:
            self._handle_generic_container(annotation, value, source)
        else:
            # Handle non-generic types
            self._handle_non_generic_annotation(annotation, value, source)
            
    def _handle_union_annotation(self, annotation: Any, value: Any, source: str):
        """Handle Union[A, B, ...] annotations."""
        args_info = get_concrete_args(annotation)
        args = [arg_info.resolved_type for arg_info in args_info]
        value_type = type(value)
        
        # Find TypeVars in the union
        typevars = {arg for arg in args if isinstance(arg, TypeVar)}
        concrete_types_in_union = {arg for arg in args if not isinstance(arg, TypeVar)}
        
        # Check if value matches any concrete type in union
        if value_type in concrete_types_in_union:
            # Value exactly matches a concrete type - no new constraints needed
            return
            
        # Try to match value against each union alternative
        best_match = None
        best_score = -1
        
        for alternative in args:
            try:
                # Try to process this alternative
                if isinstance(alternative, TypeVar):
                    # Direct TypeVar - simple match
                    score = 1
                    if score > best_score:
                        best_score = score
                        best_match = alternative
                else:
                    # Check if alternative can handle the value
                    alt_origin = get_generic_origin(alternative)
                    if alt_origin is get_generic_origin(type(value)) or alt_origin is type(value):
                        # Origins match - this is a good candidate
                        score = 10  # Higher score for structured matches
                        if score > best_score:
                            best_score = score
                            best_match = alternative
                    elif alt_origin is None and alternative is type(value):
                        # Direct type match
                        score = 5
                        if score > best_score:
                            best_score = score
                            best_match = alternative
            except (TypeError, AttributeError):
                continue
                
        if best_match is not None:
            if isinstance(best_match, TypeVar):
                # Direct TypeVar binding
                concrete_type = _infer_type_from_value(value)
                self.add_equality_constraint(best_match, concrete_type, f"{source}:union_direct", Variance.INVARIANT)
                self.add_bounds_constraint(best_match, f"{source}:union_bounds")
            else:
                # Process the matching alternative
                self._collect_constraints_recursive(best_match, value, f"{source}:union_match")
        else:
            # Fallback: if we have TypeVars and no good match, create subset constraint
            if typevars:
                self.add_subset_constraint(typevars, {value_type}, Variance.COVARIANT, f"{source}:union_fallback")
            
    def _handle_generic_container(self, annotation: Any, value: Any, source: str):
        """Unified handler for all generic container types using generic_utils and variance rules."""
        
        # Get generic information
        ann_info = self.generic_utils.get_generic_info(annotation)
        
        if not ann_info.is_generic:
            # Not actually generic, handle as non-generic
            self._handle_non_generic_annotation(annotation, value, source)
            return
        
        # Special handling for tuples
        if ann_info.origin in (tuple, Tuple):
            self._handle_tuple_annotation(annotation, value, source)
            return
        
        # Get variance rules for this container type
        variance_rules = VARIANCE_MAP.get(ann_info.origin, [Variance.INVARIANT] * len(ann_info.concrete_args))
        
        # Ensure we have enough variance rules
        while len(variance_rules) < len(ann_info.concrete_args):
            variance_rules.append(Variance.INVARIANT)
        
        # Validate value type
        if not self._validate_container_type(ann_info.origin, value):
            raise CSPTypeInferenceError(f"Expected {ann_info.origin}, got {type(value)} in {source}")
        
        # Use generic_utils to extract concrete types from the instance
        inferred_concrete_args_info = self.generic_utils.get_instance_concrete_args(value)
        
        # Create constraints for each type parameter
        for i, (type_arg_info, variance) in enumerate(zip(ann_info.concrete_args, variance_rules)):
            type_arg = type_arg_info.resolved_type
            if isinstance(type_arg, TypeVar):
                # Handle TypeVar with appropriate variance
                if i < len(inferred_concrete_args_info):
                    inferred_info = inferred_concrete_args_info[i]
                    inferred_type = inferred_info.resolved_type
                    self._add_constraint_for_typevar_with_type(type_arg, inferred_type, variance, f"{source}:{ann_info.origin.__name__}[{i}]")
            else:
                # Recursively handle nested generic structures
                if i < len(inferred_concrete_args_info):
                    # Get the actual values for this type parameter position
                    nested_annotation = type_arg_info.resolved_type
                    nested_values = self._extract_nested_values(value, ann_info.origin, i)
                    
                    # Process each nested value against the nested annotation
                    for j, nested_value in enumerate(nested_values):
                        self._collect_constraints_recursive(nested_annotation, nested_value, f"{source}:{ann_info.origin.__name__}[{i}]:nested[{j}]")
    
    def _add_constraint_for_typevar_with_type(self, typevar: TypeVar, inferred_type: type, variance: Variance, source: str):
        """Add constraints for a TypeVar based on an already-inferred type and variance."""
        
        # Check if inferred_type is a union - if so, create subset constraint instead
        origin = get_generic_origin(inferred_type)
        if origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType')):
            # Extract union members and create subset constraint
            args_info = get_concrete_args(inferred_type)
            union_types = {arg_info.resolved_type for arg_info in args_info}
            self.add_subset_constraint({typevar}, union_types, variance, source)
        else:
            # Apply variance rules for non-union types
            if variance == Variance.INVARIANT:
                # Invariant: must be exact match
                self.add_equality_constraint(typevar, inferred_type, source, variance)
            elif variance == Variance.COVARIANT:
                # Covariant: TypeVar can be inferred_type or any supertype
                # T ≥ inferred_type (T must be supertype of inferred_type)
                self.add_supertype_constraint(typevar, inferred_type, source)
            elif variance == Variance.CONTRAVARIANT:
                # Contravariant: TypeVar can be inferred_type or any subtype
                # T ≤ inferred_type (T must be subtype of inferred_type)
                self.add_subtype_constraint(typevar, inferred_type, source)
        
        # Always add bounds check
        self.add_bounds_constraint(typevar, f"{source}:bounds")
    
    def _handle_tuple_annotation(self, annotation: Any, value: Any, source: str):
        """Handle Tuple annotations with proper distinction between fixed and variable length."""
        if not isinstance(value, tuple):
            raise CSPTypeInferenceError(f"Expected tuple, got {type(value)} in {source}")
            
        args_info = get_concrete_args(annotation)
        args = [arg_info.resolved_type for arg_info in args_info]
        
        if len(args) == 2 and args[1] is ...:
            # Variable length tuple: Tuple[T, ...]
            element_annotation = args[0]
            if isinstance(element_annotation, TypeVar):
                if value:
                    element_types = {type(item) for item in value}
                    if len(element_types) == 1:
                        self.add_equality_constraint(element_annotation, next(iter(element_types)), f"{source}:var_tuple", Variance.COVARIANT)
                    else:
                        self.add_subset_constraint({element_annotation}, element_types, Variance.COVARIANT, f"{source}:var_tuple_mixed")
                    self.add_bounds_constraint(element_annotation, f"{source}:var_tuple_bounds")
            else:
                # Recursively handle each element with the same annotation
                for i, item in enumerate(value):
                    self._collect_constraints_recursive(element_annotation, item, f"{source}:var_tuple[{i}]")
        else:
            # Fixed length tuple: Tuple[X, Y, Z, ...]
            for i, item in enumerate(value):
                if i < len(args):
                    # Each position has its own type parameter
                    type_arg = args[i]
                    if isinstance(type_arg, TypeVar):
                        concrete_type = _infer_type_from_value(item)
                        self.add_equality_constraint(type_arg, concrete_type, f"{source}:fixed_tuple[{i}]", Variance.INVARIANT)
                        self.add_bounds_constraint(type_arg, f"{source}:fixed_tuple[{i}]_bounds")
                    else:
                        # Recursively handle nested types
                        self._collect_constraints_recursive(type_arg, item, f"{source}:fixed_tuple[{i}]")
    
    def _validate_container_type(self, origin: type, value: Any) -> bool:
        """Validate that value is of the expected container type."""
        if origin in (list, List):
            return isinstance(value, list)
        elif origin in (dict, Dict):
            return isinstance(value, dict)
        elif origin in (tuple, Tuple):
            return isinstance(value, tuple)
        elif origin in (set, Set):
            return isinstance(value, set)
        elif origin is Callable:
            return callable(value)
        else:
            # For custom types, check if value is instance of origin
            try:
                return isinstance(value, origin)
            except TypeError:
                # origin might not be a type we can check against
                return True
    
    def _handle_non_generic_annotation(self, annotation: Any, value: Any, source: str):
        """Handle non-generic type annotations."""
        # For non-generic types, just validate that the value matches
        expected_type = annotation
        actual_type = type(value)
        
        if not _is_subtype(actual_type, expected_type):
            # This is not necessarily an error - might be a supertype relationship
            # For now, just skip (could add warnings later)
            # Use source for potential debugging information if needed
            _ = source  # Explicitly mark as intentionally unused
    
    def solve(self) -> CSPSolution:
        """Solve the CSP to find type bindings."""
        
        if not self.domains:
            return CSPSolution()  # No TypeVars to solve
            
        # Apply constraints to refine domains
        self._propagate_constraints()
        
        # Check for unsatisfiable domains
        for typevar, domain in self.domains.items():
            if domain.is_empty():
                raise CSPTypeInferenceError(f"No valid types for {typevar} after constraint propagation")
                
        # Generate solution
        solution = CSPSolution()
        for typevar, domain in self.domains.items():
            try:
                best_type = domain.get_best_type()
                solution.bind(typevar, best_type)
            except CSPTypeInferenceError as e:
                raise CSPTypeInferenceError(f"Failed to resolve {typevar}: {e}") from e
                
        return solution
        
    def _propagate_constraints(self):
        """Propagate constraints to refine TypeVar domains."""
        
        # Sort constraints by priority (higher priority first)
        sorted_constraints = sorted(self.constraints, key=lambda c: c.priority, reverse=True)
        
        for constraint in sorted_constraints:
            try:
                self._apply_constraint(constraint)
            except Exception as e:
                raise CSPTypeInferenceError(f"Failed to apply constraint {constraint}: {e}") from e
                
    def _apply_constraint(self, constraint: TypeConstraint):
        """Apply a single constraint to refine domains."""
        
        if constraint.constraint_type == ConstraintType.EQUALITY:
            # A = type
            typevar = next(iter(constraint.variables))
            concrete_type = next(iter(constraint.types))
            
            # Check if there's already an exact type with lower priority
            domain = self.domains[typevar]
            if domain.exact_type is not None:
                # Find the priority of the existing constraint for this domain
                existing_priority = self._get_domain_priority(typevar)
                if constraint.priority > existing_priority:
                    # Higher priority constraint - override existing
                    domain.exact_type = concrete_type
                    self._set_domain_priority(typevar, constraint.priority)
                    self._set_domain_source(typevar, constraint.source)
                elif constraint.priority == existing_priority:
                    # Same priority - check if types are compatible or if we should create union
                    if domain.exact_type != concrete_type:
                        # Check if this is an explicit conflict (different manual sources) vs natural inference
                        existing_source = self._get_existing_constraint_source(typevar)
                        is_explicit_conflict = (existing_source != constraint.source and 
                                              not self._are_related_sources(existing_source, constraint.source))
                        
                        if is_explicit_conflict and self._are_incompatible_types(domain.exact_type, concrete_type):
                            raise CSPTypeInferenceError(f"Incompatible constraints for {typevar}: {domain.exact_type} vs {concrete_type}")
                        else:
                            # Create union for natural inference conflicts or compatible types
                            origin = get_generic_origin(domain.exact_type)
                            if origin is Union:
                                existing_args_info = get_concrete_args(domain.exact_type)
                                existing_args = {arg_info.resolved_type for arg_info in existing_args_info}
                                existing_args.add(concrete_type)
                                domain.exact_type = create_union_if_needed(existing_args)
                            else:
                                domain.exact_type = create_union_if_needed({domain.exact_type, concrete_type})
                # Lower priority constraint is ignored
            else:
                # No existing constraint - set the type
                domain.exact_type = concrete_type
                # Track the priority and source for this domain
                self._set_domain_priority(typevar, constraint.priority)
                self._set_domain_source(typevar, constraint.source)
            
        elif constraint.constraint_type == ConstraintType.SUBSET:
            # {TypeVars} ⊇ {concrete_types} - distribute types among TypeVars
            if len(constraint.variables) == 1:
                # Single TypeVar must be union of all types
                typevar = next(iter(constraint.variables))
                
                # Check if there's already an exact type with higher priority
                domain = self.domains[typevar]
                if domain.exact_type is not None:
                    existing_priority = self._get_domain_priority(typevar)
                    if constraint.priority <= existing_priority:
                        # Lower or equal priority constraint - don't override exact type
                        return
                
                # Apply the subset constraint
                if len(constraint.types) == 1:
                    self.domains[typevar].set_exact_type(next(iter(constraint.types)))
                else:
                    union_type = create_union_if_needed(constraint.types)
                    self.domains[typevar].set_exact_type(union_type)
                self._set_domain_priority(typevar, constraint.priority)
            else:
                # Multiple TypeVars - for now, add all types as possibilities to all vars
                # More sophisticated assignment could be added here
                for typevar in constraint.variables:
                    for t in constraint.types:
                        self.domains[typevar].add_possible_type(t)
                        
        elif constraint.constraint_type == ConstraintType.SUBTYPE:
            # A ≤ SuperType (covariant)
            typevar = next(iter(constraint.variables))
            supertype = next(iter(constraint.types))
            self.domains[typevar].add_subtype_constraint(supertype)
            
        elif constraint.constraint_type == ConstraintType.SUPERTYPE:
            # A ≥ SubType (contravariant)
            typevar = next(iter(constraint.variables))
            subtype = next(iter(constraint.types))
            domain = self.domains[typevar]
            domain.add_supertype_constraint(subtype)
            
            # For covariant constraints, also add the observed type and some common supertypes
            # This gives the domain a base set of types to work with
            domain.add_possible_type(subtype)  # The observed type itself
            domain.add_possible_type(object)   # Universal supertype
            
            # Add some common supertypes for built-in types
            if subtype is int:
                domain.add_possible_type(float)  # int can be considered as float in some contexts
            elif subtype in (int, float):
                domain.add_possible_type(object)  # numbers are objects
            
        elif constraint.constraint_type == ConstraintType.BOUNDS_CHECK:
            # Check TypeVar bounds and constraints
            typevar = next(iter(constraint.variables))
            self._apply_typevar_bounds(typevar)
            
    def _apply_typevar_bounds(self, typevar: TypeVar):
        """Apply TypeVar bounds and constraints to its domain."""
        domain = self.domains[typevar]
        
        # Apply explicit constraints (e.g., TypeVar('T', int, str))
        if typevar.__constraints__:
            # TypeVar can only be one of the constraint types
            valid_types = set(typevar.__constraints__)
            if domain.exact_type is not None:
                # Check if exact type satisfies constraints
                if domain.exact_type not in valid_types:
                    # For Union types, check if all components satisfy constraints
                    origin = get_generic_origin(domain.exact_type)
                    if origin is Union:
                        union_args_info = get_concrete_args(domain.exact_type)
                        union_args = [arg_info.resolved_type for arg_info in union_args_info]
                        if not all(arg in valid_types for arg in union_args):
                            raise CSPTypeInferenceError(f"Type {domain.exact_type} violates constraints {valid_types}")
                    else:
                        raise CSPTypeInferenceError(f"Type {domain.exact_type} violates constraints {valid_types}")
            else:
                # Restrict domain to constraint types
                domain.possible_types &= valid_types
                
        # Apply bound (e.g., TypeVar('T', bound=int))
        if typevar.__bound__:
            domain.add_subtype_constraint(typevar.__bound__)
    
    def _get_domain_priority(self, typevar: TypeVar) -> int:
        """Get the priority of the current constraint for a TypeVar domain."""
        return self.domain_priorities.get(typevar, 0)
    
    def _set_domain_priority(self, typevar: TypeVar, priority: int):
        """Set the priority for a TypeVar domain."""
        self.domain_priorities[typevar] = priority
    
    def _get_existing_constraint_source(self, typevar: TypeVar) -> str:
        """Get the source of the existing constraint for a TypeVar domain."""
        return self.domain_sources.get(typevar, "")
    
    def _set_domain_source(self, typevar: TypeVar, source: str):
        """Set the source for a TypeVar domain."""
        self.domain_sources[typevar] = source
    
    def _are_related_sources(self, source1: str, source2: str) -> bool:
        """Check if two constraint sources are related (from same analysis)."""
        # Sources are related if they have the same prefix (before the first ':')
        if not source1 or not source2:
            return True  # Empty sources are considered related
        
        prefix1 = source1.split(':')[0] 
        prefix2 = source2.split(':')[0]
        
        # Function parameters (param_*) are all related
        if prefix1.startswith('param_') and prefix2.startswith('param_'):
            return True
        
        # Keyword arguments (kwarg_*) are all related
        if prefix1.startswith('kwarg_') and prefix2.startswith('kwarg_'):
            return True
        
        # Same prefix are related
        return prefix1 == prefix2
    
    def _are_incompatible_types(self, type1: type, type2: type) -> bool:
        """Check if two types are fundamentally incompatible and cannot be unified."""
        # Basic built-in types that don't have inheritance relationships are incompatible
        basic_types = {int, str, float, bool, bytes, type(None)}
        
        # Both are basic types and different - incompatible
        if type1 in basic_types and type2 in basic_types and type1 != type2:
            # Special case: bool is a subtype of int, so they're compatible
            if {type1, type2} == {bool, int}:
                return False
            return True
        
        # If either is a union, they might be compatible
        origin1 = get_generic_origin(type1)
        origin2 = get_generic_origin(type2)
        if origin1 is Union or origin2 is Union:
            return False
        
        # Generic types with same origin but different args might be compatible
        if origin1 and origin2 and origin1 == origin2:
            return False
        
        # For complex types, default to compatible (create union)
        return False
    
    def _extract_nested_values(self, instance: Any, container_origin: type, position: int) -> List[Any]:
        """Extract nested values from an instance based on container type and type parameter position."""
        if container_origin in (dict, Dict):
            if position == 0:
                # Keys of the dict
                return list(instance.keys())
            elif position == 1:
                # Values of the dict
                return list(instance.values())
        elif container_origin in (list, List):
            if position == 0:
                # Elements of the list
                return list(instance)
        elif container_origin in (tuple, Tuple):
            if position == 0:
                # Elements of the tuple (assuming variable-length tuple)
                return list(instance)
        elif container_origin in (set, Set):
            if position == 0:
                # Elements of the set
                return list(instance)
        elif container_origin is Callable:
            # For Callable types, we can't extract types from callable instances at runtime
            # without inspecting the function signature. This is a limitation of runtime type extraction.
            # Return empty list to avoid creating constraints for now.
            return []
        
        # Default: return empty list if we can't extract values
        return []


def infer_return_type_csp(
    fn: callable,
    *args,
    type_overrides: Optional[Dict[TypeVar, type]] = None,
    **kwargs,
) -> type:
    """
    Infer the concrete return type using CSP-based algorithm.
    
    This treats type inference as a constraint satisfaction problem.
    """
    
    if type_overrides is None:
        type_overrides = {}
        
    # Get function signature and return annotation
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation
    
    if return_annotation is inspect.Signature.empty:
        raise ValueError("Function must have return type annotation")
        
    # Create CSP engine
    engine = CSPTypeInferenceEngine()
    
    # Collect constraints from function parameters
    param_names = list(sig.parameters.keys())
    
    # Process positional arguments
    for i, arg in enumerate(args):
        if i < len(param_names):
            param = sig.parameters[param_names[i]]
            if param.annotation != inspect.Parameter.empty:
                engine.collect_constraints_from_annotation_value(
                    param.annotation, arg, f"param_{param.name}"
                )
                
    # Process keyword arguments
    for name, value in kwargs.items():
        if name in sig.parameters:
            param = sig.parameters[name]
            if param.annotation != inspect.Parameter.empty:
                engine.collect_constraints_from_annotation_value(
                    param.annotation, value, f"kwarg_{name}"
                )
                
    # Add type overrides as high-priority equality constraints
    for typevar, override_type in type_overrides.items():
        engine.add_equality_constraint(typevar, override_type, "override", Variance.INVARIANT)
        
    # Solve the CSP
    try:
        solution = engine.solve()
    except CSPTypeInferenceError as e:
        raise CSPTypeInferenceError(f"CSP solving failed: {e}") from e
        
    # Apply solution to return annotation
    result = solution.apply(return_annotation)
    
    # Check for any remaining unbound TypeVars
    if _has_unbound_typevars(result):
        unbound = _find_unbound_typevars(result)
        raise CSPTypeInferenceError(f"Could not infer types for: {unbound}")
        
    return result


# =============================================================================
# Helper functions (updated to use generic_utils)
# =============================================================================

def _is_subtype(subtype: type, supertype: type) -> bool:
    """Check if subtype is a subtype of supertype."""
    try:
        return issubclass(subtype, supertype)
    except TypeError:
        return False


def _infer_type_from_value(value: Any) -> type:
    """Infer the most specific type from a value."""
    if value is None:
        return type(None)
    return type(value)


def _substitute_typevars(annotation: Any, bindings: Dict[TypeVar, type]) -> Any:
    """Substitute TypeVars in an annotation with their bindings."""
    
    if isinstance(annotation, TypeVar):
        return bindings.get(annotation, annotation)
        
    # Use generic_utils for consistent type handling
    origin = get_generic_origin(annotation)
    args_info = get_concrete_args(annotation)
    
    if not origin or not args_info:
        return annotation
    
    # Extract raw types from GenericInfo objects
    args = [arg_info.resolved_type for arg_info in args_info]
    
    # Handle Union types specially - filter out unbound TypeVars (like unification system)
    if origin is Union:
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
            return create_union_if_needed(set(substituted_args))
        
        # If no args were bound, return the original annotation (will be caught as unbound)
        return annotation
        
    # Recursively substitute in type arguments
    substituted_args = []
    for arg in args:
        substituted_args.append(_substitute_typevars(arg, bindings))
        
    # Reconstruct the type
    if origin in (list, List):
        return list[substituted_args[0]]
    elif origin in (dict, Dict):
        return dict[substituted_args[0], substituted_args[1]]
    elif origin in (tuple, Tuple):
        return tuple[tuple(substituted_args)]
    elif origin in (set, Set):
        return set[substituted_args[0]]
    elif origin is Union:
        if len(substituted_args) == 1:
            return substituted_args[0]
        return create_union_if_needed(set(substituted_args))
    else:
        # For other generic types, try to reconstruct
        try:
            return origin[tuple(substituted_args)]
        except (TypeError, AttributeError):
            return annotation


def _has_unbound_typevars(annotation: Any) -> bool:
    """Check if an annotation contains any unbound TypeVars."""
    if isinstance(annotation, TypeVar):
        return True
        
    # Use generic_utils for consistent handling
    args_info = get_concrete_args(annotation)
    
    if args_info:
        args = [arg_info.resolved_type for arg_info in args_info]
        return any(_has_unbound_typevars(arg) for arg in args)
        
    return False


def _find_unbound_typevars(annotation: Any) -> Set[TypeVar]:
    """Find all unbound TypeVars in an annotation."""
    if isinstance(annotation, TypeVar):
        return {annotation}
        
    result = set()
    # Use generic_utils for consistent handling
    args_info = get_concrete_args(annotation)
    
    if args_info:
        args = [arg_info.resolved_type for arg_info in args_info]
        for arg in args:
            result.update(_find_unbound_typevars(arg))
            
    return result
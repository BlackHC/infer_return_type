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
import typing
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union, get_origin, get_args
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
import types

# Import unified generic utilities
from generic_utils import (
    GenericTypeUtils, get_generic_info, get_instance_generic_info, 
    get_type_parameters, get_concrete_args, get_instance_concrete_args,
    get_generic_origin, is_generic_type, extract_all_typevars, create_union_if_needed
)


class CSPTypeInferenceError(Exception):
    """Raised when CSP-based type inference fails."""
    pass


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
    # typing.Callable: [Variance.CONTRAVARIANT, Variance.COVARIANT], # Callable[[T], R] - contravariant in T, covariant in R
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
        return f"{self.constraint_type.value}: vars={{{vars_str}}} types={{{types_str}}} - {self.description} ({self.variance.value})"


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
        """Set the exact type (strongest constraint). If there's already an exact type, create union."""
        if self.exact_type is None:
            self.exact_type = t
        else:
            # Conflict! Need to create union
            if self.exact_type != t:
                # Create union of existing and new type
                origin = get_generic_origin(self.exact_type)
                if origin is Union:
                    # Already a union - add to it
                    existing_args = set(get_concrete_args(self.exact_type))
                    existing_args.add(t)
                    self.exact_type = create_union_if_needed(existing_args)
                else:
                    # Create new union
                    self.exact_type = create_union_if_needed({self.exact_type, t})
        
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
        
        # Multiple valid types - create union
        return create_union_if_needed(valid)


class CSPTypeInferenceEngine:
    """Main CSP solver for type inference."""
    
    def __init__(self):
        self.constraints: List[TypeConstraint] = []
        self.domains: Dict[TypeVar, TypeDomain] = {}
        self.solutions: List[CSPSolution] = []
        # Use unified generic type utilities
        self.generic_utils = GenericTypeUtils()
        
    def clear(self):
        """Clear all constraints and domains for fresh solving."""
        self.constraints.clear()
        self.domains.clear()
        self.solutions.clear()
        
    def add_constraint(self, constraint: TypeConstraint):
        """Add a constraint to the CSP."""
        self.constraints.append(constraint)
        
        # Initialize domains for any new TypeVars
        for var in constraint.variables:
            if var not in self.domains:
                self.domains[var] = TypeDomain(var)
                
    def add_equality_constraint(self, typevar: TypeVar, concrete_type: type, variance: Variance = Variance.INVARIANT, source: str = ""):
        """Add A = type constraint."""
        constraint = TypeConstraint(
            constraint_type=ConstraintType.EQUALITY,
            variables={typevar},
            types={concrete_type},
            description=f"{typevar} = {concrete_type}",
            priority=10,  # High priority - exact constraints
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
            self.add_equality_constraint(annotation, concrete_type, Variance.INVARIANT, f"{source}:direct")
            self.add_bounds_constraint(annotation, f"{source}:bounds")
            return
            
        # Use generic_utils for consistent type information extraction
        origin = get_generic_origin(annotation)
        args = get_concrete_args(annotation)
        
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
        if origin and args:
            self._handle_generic_container(annotation, value, source)
        else:
            # Handle non-generic types
            self._handle_non_generic_annotation(annotation, value, source)
            
    def _handle_union_annotation(self, annotation: Any, value: Any, source: str):
        """Handle Union[A, B, ...] annotations."""
        args = get_concrete_args(annotation)
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
                    if alt_origin == get_generic_origin(type(value)) or alt_origin == type(value):
                        # Origins match - this is a good candidate
                        score = 10  # Higher score for structured matches
                        if score > best_score:
                            best_score = score
                            best_match = alternative
                    elif alt_origin is None and alternative == type(value):
                        # Direct type match
                        score = 5
                        if score > best_score:
                            best_score = score
                            best_match = alternative
            except Exception:
                continue
                
        if best_match is not None:
            if isinstance(best_match, TypeVar):
                # Direct TypeVar binding
                concrete_type = _infer_type_from_value(value)
                self.add_equality_constraint(best_match, concrete_type, Variance.INVARIANT, f"{source}:union_direct")
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
        inferred_concrete_args = self.generic_utils.get_instance_concrete_args(value)
        
        # Create constraints for each type parameter
        for i, (type_arg, variance) in enumerate(zip(ann_info.concrete_args, variance_rules)):
            if isinstance(type_arg, TypeVar):
                # Handle TypeVar with appropriate variance
                if i < len(inferred_concrete_args):
                    inferred_type = inferred_concrete_args[i]
                    self._add_constraint_for_typevar_with_type(type_arg, inferred_type, variance, f"{source}:{ann_info.origin.__name__}[{i}]")
            else:
                # Recursively handle nested generic structures
                # For non-TypeVar annotations, we would need to dive deeper, but this is rare
                pass
    
    def _add_constraint_for_typevar_with_type(self, typevar: TypeVar, inferred_type: type, variance: Variance, source: str):
        """Add constraints for a TypeVar based on an already-inferred type and variance."""
        
        # Apply variance rules
        if variance == Variance.INVARIANT:
            # Invariant: must be exact match
            self.add_equality_constraint(typevar, inferred_type, variance, source)
        elif variance == Variance.COVARIANT:
            # Covariant: TypeVar can be the inferred type or a union containing it
            self.add_equality_constraint(typevar, inferred_type, variance, source)
        elif variance == Variance.CONTRAVARIANT:
            # Contravariant: TypeVar must be supertype of inferred type
            self.add_supertype_constraint(typevar, inferred_type, source)
        
        # Always add bounds check
        self.add_bounds_constraint(typevar, f"{source}:bounds")
    
    def _handle_tuple_annotation(self, annotation: Any, value: Any, source: str):
        """Handle Tuple annotations with proper distinction between fixed and variable length."""
        if not isinstance(value, tuple):
            raise CSPTypeInferenceError(f"Expected tuple, got {type(value)} in {source}")
            
        args = get_concrete_args(annotation)
        
        if len(args) == 2 and args[1] is ...:
            # Variable length tuple: Tuple[T, ...]
            element_annotation = args[0]
            if isinstance(element_annotation, TypeVar):
                if value:
                    element_types = {type(item) for item in value}
                    if len(element_types) == 1:
                        self.add_equality_constraint(element_annotation, next(iter(element_types)), Variance.COVARIANT, f"{source}:var_tuple")
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
                        self.add_equality_constraint(type_arg, concrete_type, Variance.INVARIANT, f"{source}:fixed_tuple[{i}]")
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
            pass
    
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
                raise CSPTypeInferenceError(f"Failed to resolve {typevar}: {e}")
                
        return solution
        
    def _propagate_constraints(self):
        """Propagate constraints to refine TypeVar domains."""
        
        # Sort constraints by priority (higher priority first)
        sorted_constraints = sorted(self.constraints, key=lambda c: c.priority, reverse=True)
        
        for constraint in sorted_constraints:
            try:
                self._apply_constraint(constraint)
            except Exception as e:
                raise CSPTypeInferenceError(f"Failed to apply constraint {constraint}: {e}")
                
    def _apply_constraint(self, constraint: TypeConstraint):
        """Apply a single constraint to refine domains."""
        
        if constraint.constraint_type == ConstraintType.EQUALITY:
            # A = type
            typevar = next(iter(constraint.variables))
            concrete_type = next(iter(constraint.types))
            
            # Apply variance rules when setting exact type
            if constraint.variance == Variance.COVARIANT and len(constraint.types) > 1:
                # Covariant with multiple types - create union
                union_type = create_union_if_needed(constraint.types)
                self.domains[typevar].set_exact_type(union_type)
            else:
                self.domains[typevar].set_exact_type(concrete_type)
            
        elif constraint.constraint_type == ConstraintType.SUBSET:
            # {TypeVars} ⊇ {concrete_types} - distribute types among TypeVars
            if len(constraint.variables) == 1:
                # Single TypeVar must be union of all types
                typevar = next(iter(constraint.variables))
                if len(constraint.types) == 1:
                    if constraint.variance == Variance.COVARIANT:
                        self.domains[typevar].set_exact_type(next(iter(constraint.types)))
                    else:
                        self.domains[typevar].set_exact_type(next(iter(constraint.types)))
                else:
                    union_type = create_union_if_needed(constraint.types)
                    self.domains[typevar].set_exact_type(union_type)
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
            self.domains[typevar].add_supertype_constraint(subtype)
            
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
                        union_args = get_concrete_args(domain.exact_type)
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
        engine.add_equality_constraint(typevar, override_type, Variance.INVARIANT, "override")
        
    # Solve the CSP
    try:
        solution = engine.solve()
    except CSPTypeInferenceError as e:
        raise CSPTypeInferenceError(f"CSP solving failed: {e}")
        
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
    args = get_concrete_args(annotation)
    
    if not origin or not args:
        return annotation
    
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
        except Exception:
            return annotation


def _has_unbound_typevars(annotation: Any) -> bool:
    """Check if an annotation contains any unbound TypeVars."""
    if isinstance(annotation, TypeVar):
        return True
        
    # Use generic_utils for consistent handling
    origin = get_generic_origin(annotation)
    args = get_concrete_args(annotation)
    
    if args:
        return any(_has_unbound_typevars(arg) for arg in args)
        
    return False


def _find_unbound_typevars(annotation: Any) -> Set[TypeVar]:
    """Find all unbound TypeVars in an annotation."""
    if isinstance(annotation, TypeVar):
        return {annotation}
        
    result = set()
    # Use generic_utils for consistent handling
    origin = get_generic_origin(annotation)
    args = get_concrete_args(annotation)
    
    if args:
        for arg in args:
            result.update(_find_unbound_typevars(arg))
            
    return result
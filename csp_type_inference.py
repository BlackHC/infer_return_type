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
    GenericTypeUtils, get_concrete_args, get_generic_origin, create_union_if_needed,
    get_annotation_value_pairs, is_union_type
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
            # Check if exact type satisfies TypeVar bounds/constraints
            candidate = {self.exact_type}
            
            # Apply TypeVar bounds and constraints (these are fundamental type system rules)
            if self.typevar.__constraints__:
                # TypeVar can only be one of the constraint types
                if self.exact_type not in self.typevar.__constraints__:
                    return set()  # Invalid
            
            if self.typevar.__bound__:
                # TypeVar must be subtype of bound
                if not _is_subtype(self.exact_type, self.typevar.__bound__):
                    return set()  # Invalid
            
            # For normal inference constraints (subtype/supertype), be more lenient
            # to allow type overrides to work
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
        # Fallback evidence for unbound TypeVars
        self._fallback_evidence: List[Tuple[type, str]] = []
        # Use unified generic type utilities
        self.generic_utils = GenericTypeUtils()
        
    def clear(self):
        """Clear all constraints and domains for fresh solving."""
        self.constraints.clear()
        self.domains.clear()
        self.solutions.clear()
        self.domain_priorities.clear()
        self.domain_sources.clear()
        self._fallback_evidence.clear()
        
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
        if source == "override":
            priority = 15
        elif "dict" in source.lower():
            # Dict constraints have higher priority due to structured key-value mapping
            priority = 12  
        elif any(structured in source.lower() for structured in ["tuple", "list"]):
            # Other structured containers get medium-high priority
            priority = 11
        else:
            # Default priority for other constraints
            priority = 10
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
        # Lower priority for subset constraints, especially from sets with unions
        if "set" in source.lower() and any("union" in s for s in [source]):
            priority = 3  # Very low priority for ambiguous set unions
        elif "set" in source.lower():
            priority = 4  # Low priority for sets
        else:
            priority = 5  # Default subset priority
            
        constraint = TypeConstraint(
            constraint_type=ConstraintType.SUBSET,
            variables=typevars,
            types=concrete_types,
            description=f"{{{', '.join(str(v) for v in typevars)}}} ⊇ {{{', '.join(str(t) for t in concrete_types)}}}",
            priority=priority,
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
        initial_constraint_count = len(self.constraints)
        
        self._collect_constraints_recursive(annotation, value, source)
        
        # Fallback: if no constraints were generated but the instance has rich type information,
        # try to extract concrete types from the instance and bind them to any unbound TypeVars
        # This handles cases like List[Box[A]] -> List[Box] where TypeVar A is lost in the annotation
        # but the instance [Box[int](item=1)] still contains the concrete type information
        if len(self.constraints) == initial_constraint_count:  # No new constraints generated
            self._fallback_instance_type_extraction(annotation, value, source)
        
    def _collect_constraints_recursive(self, annotation: Any, value: Any, source: str):
        """Recursively collect constraints from annotation/value pairs."""
        
        # Base case: Direct TypeVar
        if isinstance(annotation, TypeVar):
            concrete_type = _infer_type_from_value(value)
            self.add_equality_constraint(annotation, concrete_type, f"{source}:direct", Variance.INVARIANT)
            self.add_bounds_constraint(annotation, f"{source}:bounds")
            return
            
        # Handle Union types (including Optional) first
        origin = get_generic_origin(annotation)
        if is_union_type(origin):
            self._handle_union_annotation(annotation, value, source)
            return
            
        # Use the new get_annotation_value_pairs function for unified constraint extraction
        try:
            pairs = get_annotation_value_pairs(annotation, value)
            
            if pairs:
                # Group pairs by TypeVar to handle them properly
                typevar_value_groups = {}
                
                for generic_info, val in pairs:
                    # Handle direct TypeVars
                    if isinstance(generic_info.origin, TypeVar):
                        typevar = generic_info.origin
                        if typevar not in typevar_value_groups:
                            typevar_value_groups[typevar] = []
                        
                        # Improved: Extract concrete types from complex values
                        concrete_types = self._extract_concrete_types_from_value(val)
                        typevar_value_groups[typevar].extend(concrete_types)
                        
                    # Handle Union types - need to determine which TypeVar the value should bind to
                    elif is_union_type(generic_info.origin):
                        self._handle_union_value_binding(generic_info, val, typevar_value_groups)
                    else:
                        # Extract TypeVars from nested concrete_args
                        self._extract_typevars_from_generic_info(generic_info, val, typevar_value_groups)
                        
                        # Only use recursive processing if we didn't extract TypeVars from concrete_args
                        # This prevents double-processing the same data
                        if not self._has_typevars_in_concrete_args(generic_info):
                            self._collect_constraints_recursive(generic_info.resolved_type, val, f"{source}:nested")
                
                # Create constraints for each TypeVar
                for typevar, value_types in typevar_value_groups.items():
                    if len(value_types) == 1:
                        # Single type - create equality constraint
                        self.add_equality_constraint(typevar, value_types[0], f"{source}:unified", Variance.COVARIANT)
                    elif len(value_types) > 1:
                        # Multiple types - create subset constraint (union)
                        unique_types = set(value_types)
                        if len(unique_types) == 1:
                            # All the same type, even if multiple values
                            self.add_equality_constraint(typevar, next(iter(unique_types)), f"{source}:unified", Variance.COVARIANT)
                        else:
                            # Different types - create subset constraint
                            self.add_subset_constraint({typevar}, unique_types, Variance.COVARIANT, f"{source}:unified_mixed")
                    
                    # Always add bounds constraints
                    self.add_bounds_constraint(typevar, f"{source}:bounds")
                
                return
            
        except (TypeError, AttributeError, ValueError):
            # Fall back to the old approach if the new one fails
            pass
            
        # Fallback: Handle using the old method for complex cases
        self._handle_annotation_fallback(annotation, value, source)
    
    def _extract_typevars_from_generic_info(self, generic_info, instance, typevar_value_groups):
        """Extract TypeVars from nested concrete_args and map them to concrete types from the instance."""
        
        # Primary approach: Get the concrete type information from the instance
        instance_generic_info = self.generic_utils.get_instance_generic_info(instance)
        
        # If the instance has concrete type information, use it to bind TypeVars directly
        if instance_generic_info.concrete_args:
            # Map between annotation TypeVars and instance concrete types
            self._map_annotation_to_instance_typevars(
                generic_info.concrete_args, 
                instance_generic_info.concrete_args, 
                typevar_value_groups
            )
            return  # Direct mapping worked, we're done
        
        # Fallback: Try to get more detailed pairs by calling get_annotation_value_pairs recursively
        # This handles cases like Tuple[Dict[A, B], Set[A | B]] by drilling down into the tuple elements
        # Only use this when direct mapping didn't work
        try:
            nested_pairs = get_annotation_value_pairs(generic_info.resolved_type, instance)
            if nested_pairs:
                # Process the nested pairs through the main constraint collection logic
                for nested_generic_info, nested_value in nested_pairs:
                    if isinstance(nested_generic_info.origin, TypeVar):
                        typevar = nested_generic_info.origin
                        if typevar not in typevar_value_groups:
                            typevar_value_groups[typevar] = []
                        # Use the improved extraction for these nested values
                        concrete_types = self._extract_concrete_types_from_value(nested_value)
                        typevar_value_groups[typevar].extend(concrete_types)
                    elif is_union_type(nested_generic_info.origin):
                        self._handle_union_value_binding(nested_generic_info, nested_value, typevar_value_groups)
                    else:
                        # Recursively process even deeper
                        self._extract_typevars_from_generic_info(nested_generic_info, nested_value, typevar_value_groups)
                return  # We processed the nested pairs, so we're done
        except (TypeError, AttributeError, ValueError):
            # Fall back to the final approach if recursive processing fails
            pass
            
        # Final fallback: just collect TypeVars from annotation without concrete bindings
        self._collect_typevars_from_args(generic_info.concrete_args, typevar_value_groups, instance)
    
    def _map_annotation_to_instance_typevars(self, annotation_args, instance_args, typevar_value_groups):
        """Map TypeVars from annotation to concrete types from instance."""
        
        # Pair up annotation args with instance args
        for ann_arg, inst_arg in zip(annotation_args, instance_args):
            if isinstance(ann_arg.origin, TypeVar):
                # This is a TypeVar in the annotation - bind it to the instance type
                typevar = ann_arg.origin
                concrete_type = inst_arg.resolved_type
                
                if typevar not in typevar_value_groups:
                    typevar_value_groups[typevar] = []
                typevar_value_groups[typevar].append(concrete_type)
            elif is_union_type(ann_arg.origin):
                # Handle union types in annotation args
                # Currently handled by the recursive processing approach
                # This case is kept for completeness but union handling
                # is more effectively done in the recursive _extract_typevars_from_generic_info
                pass
            elif ann_arg.concrete_args and inst_arg.concrete_args:
                # Recursively map nested generic args
                self._map_annotation_to_instance_typevars(
                    ann_arg.concrete_args, 
                    inst_arg.concrete_args, 
                    typevar_value_groups
                )
    
    def _collect_typevars_from_args(self, args, typevar_value_groups, fallback_instance):
        """Collect TypeVars from args, using fallback instance for type inference."""
        
        for i, arg in enumerate(args):
            if isinstance(arg.origin, TypeVar):
                typevar = arg.origin
                
                # Extract concrete types from the fallback instance based on position
                concrete_types = self._extract_types_from_instance_by_position(fallback_instance, i)
                
                if typevar not in typevar_value_groups:
                    typevar_value_groups[typevar] = []
                typevar_value_groups[typevar].extend(concrete_types)
            elif arg.concrete_args:
                # For nested generic structures, extract the values at this position 
                # and recursively process them against the nested annotation
                if isinstance(fallback_instance, dict) and i < 2:
                    if i == 0:
                        # Keys of the dict - extract their types directly
                        for key in fallback_instance.keys():
                            self._collect_typevars_from_args(arg.concrete_args, typevar_value_groups, key)
                    elif i == 1:
                        # Values of the dict - recursively process them
                        for value in fallback_instance.values():
                            self._collect_typevars_from_args(arg.concrete_args, typevar_value_groups, value)
                elif isinstance(fallback_instance, (list, tuple, set)) and i == 0:
                    # Elements of the container
                    for element in fallback_instance:
                        self._collect_typevars_from_args(arg.concrete_args, typevar_value_groups, element)
                else:
                    # Fallback: recursively process with the same instance
                    self._collect_typevars_from_args(arg.concrete_args, typevar_value_groups, fallback_instance)
    
    def _extract_types_from_instance_by_position(self, instance, position):
        """Extract concrete types from an instance based on the type parameter position."""
        
        if isinstance(instance, dict):
            if position == 0:
                # First type parameter - extract key types
                return [type(key) for key in instance.keys()]
            elif position == 1:
                # Second type parameter - extract value types
                return [type(value) for value in instance.values()]
        elif isinstance(instance, (list, tuple, set)):
            if position == 0:
                # First (and usually only) type parameter - extract element types
                return [type(item) for item in instance]
        
        # Fallback: use the instance type itself
        return [_infer_type_from_value(instance)]
    
    def _has_typevars_in_concrete_args(self, generic_info):
        """Check if a GenericInfo has TypeVars in its concrete_args hierarchy."""
        return self._contains_typevars_recursive(generic_info.concrete_args)
    
    def _contains_typevars_recursive(self, args):
        """Recursively check if any args contain TypeVars."""
        for arg in args:
            if isinstance(arg.origin, TypeVar):
                return True
            if arg.concrete_args and self._contains_typevars_recursive(arg.concrete_args):
                return True
        return False
    
    def _handle_union_value_binding(self, generic_info, val, typevar_value_groups):
        """Handle binding a value to a union type like Union[A, B]."""
        
        # Get the union arguments (the TypeVars in the union)
        args_info = get_concrete_args(generic_info.resolved_type)
        union_typevars = []
        union_concrete_types = []
        
        for arg_info in args_info:
            if isinstance(arg_info.origin, TypeVar):
                union_typevars.append(arg_info.origin)
            else:
                union_concrete_types.append(arg_info.resolved_type)
        
        if not union_typevars:
            return  # No TypeVars in the union
        
        # Extract concrete types from the value
        concrete_types = self._extract_concrete_types_from_value(val)
        
        # For complex unions (like Union[A, Dict[str, JsonValue[A]], List[JsonValue[A]]]),
        # we need to check which part of the union the value matches and extract accordingly
        val_type = _infer_type_from_value(val)
        
        # First check if the value matches any of the concrete types in the union
        matched_concrete = False
        for concrete_type in union_concrete_types:
            concrete_origin = get_generic_origin(concrete_type) or concrete_type
            if val_type == concrete_origin or isinstance(val, concrete_origin):
                matched_concrete = True
                # Value matches a concrete type - try to extract TypeVars from the matched structure
                if hasattr(concrete_type, '__args__') or get_concrete_args(concrete_type):
                    # This concrete type has TypeVars in it - extract from the value
                    self._extract_typevars_from_concrete_type_match(concrete_type, val, union_typevars, typevar_value_groups)
                break
        
        # If the value didn't match any concrete type in the union, 
        # it should be bound directly to one of the TypeVars
        if not matched_concrete:
            # Smart binding strategy: For Union[A, B], try to bind the value to the TypeVar
            # that already has compatible types, rather than adding to all TypeVars.
            # This prevents Union[A, B] from making both A and B into unions.
            
            best_matches = []
            
            # Check which TypeVars already have values that are compatible with val_type
            for typevar in union_typevars:
                existing_types = typevar_value_groups.get(typevar, [])
                if existing_types:
                    # If the TypeVar already has values of the same type, prefer it
                    if val_type in existing_types:
                        best_matches.append((typevar, 10))  # High priority for exact match
                    elif any(_types_are_compatible(val_type, existing_type) for existing_type in existing_types):
                        best_matches.append((typevar, 5))   # Medium priority for compatible types
                else:
                    # TypeVar has no existing bindings - medium priority
                    # For List[Union[A, B]] we want to distribute different types to different TypeVars
                    best_matches.append((typevar, 6))
            
            if best_matches:
                # Sort by priority and bind to the best match(es)
                best_matches.sort(key=lambda x: x[1], reverse=True)
                best_priority = best_matches[0][1]
                
                # For exact matches (priority 10), bind only to those TypeVars
                if best_priority == 10:
                    for typevar, priority in best_matches:
                        if priority == best_priority:
                            if typevar not in typevar_value_groups:
                                typevar_value_groups[typevar] = []
                            typevar_value_groups[typevar].extend(concrete_types)
                else:
                    # For lower priority matches, use a smart distribution strategy
                    # If we have multiple TypeVars and multiple types, try to distribute them
                    self._distribute_types_among_typevars(union_typevars, concrete_types, typevar_value_groups)
    
    def _extract_typevars_from_concrete_type_match(self, concrete_type: Any, value: Any, union_typevars: List[TypeVar], typevar_value_groups: Dict[TypeVar, List[type]]):
        """Extract TypeVars when a value matches a specific concrete type in a union."""
        # This handles cases like Union[A, Dict[str, JsonValue[A]], List[JsonValue[A]]]
        # where the value is a dict that matches Dict[str, JsonValue[A]]
        
        try:
            # Use get_annotation_value_pairs to extract from the matched structure
            pairs = get_annotation_value_pairs(concrete_type, value)
            
            for generic_info, nested_value in pairs:
                if isinstance(generic_info.origin, TypeVar) and generic_info.origin in union_typevars:
                    typevar = generic_info.origin
                    if typevar not in typevar_value_groups:
                        typevar_value_groups[typevar] = []
                    
                    # Extract concrete types from the nested value
                    concrete_types = self._extract_concrete_types_from_value(nested_value)
                    typevar_value_groups[typevar].extend(concrete_types)
                    
                elif is_union_type(generic_info.origin):
                    # Recursively handle nested unions
                    self._handle_union_value_binding(generic_info, nested_value, typevar_value_groups)
                else:
                    # Drill deeper into the structure
                    self._extract_typevars_from_generic_info(generic_info, nested_value, typevar_value_groups)
        
        except (TypeError, AttributeError, ValueError):
            # Fallback: just extract what we can from the value directly
            concrete_types = self._extract_concrete_types_from_value(value)
            if concrete_types and union_typevars:
                # Bind to the first available TypeVar as a fallback
                first_typevar = union_typevars[0]
                if first_typevar not in typevar_value_groups:
                    typevar_value_groups[first_typevar] = []
                typevar_value_groups[first_typevar].extend(concrete_types)
    
    def _distribute_types_among_typevars(self, union_typevars: List[TypeVar], concrete_types: List[type], typevar_value_groups: Dict[TypeVar, List[type]]):
        """Intelligently distribute concrete types among TypeVars in a union."""
        
        # Group concrete types by their actual type
        type_groups = {}
        for concrete_type in concrete_types:
            if concrete_type not in type_groups:
                type_groups[concrete_type] = []
            type_groups[concrete_type].append(concrete_type)
        
        unique_types = list(type_groups.keys())
        
        # Strategy: Try to assign different types to different TypeVars
        # This works well for cases like List[Union[A, B]] with mixed int/str values
        
        if len(unique_types) <= len(union_typevars):
            # We have enough TypeVars for each unique type
            for i, unique_type in enumerate(unique_types):
                if i < len(union_typevars):
                    typevar = union_typevars[i]
                    if typevar not in typevar_value_groups:
                        typevar_value_groups[typevar] = []
                    typevar_value_groups[typevar].append(unique_type)
        else:
            # More types than TypeVars - distribute as evenly as possible
            types_per_var = len(unique_types) // len(union_typevars)
            remainder = len(unique_types) % len(union_typevars)
            
            type_index = 0
            for i, typevar in enumerate(union_typevars):
                if typevar not in typevar_value_groups:
                    typevar_value_groups[typevar] = []
                
                # Assign types_per_var types to this TypeVar
                count = types_per_var + (1 if i < remainder else 0)
                for _ in range(count):
                    if type_index < len(unique_types):
                        typevar_value_groups[typevar].append(unique_types[type_index])
                        type_index += 1
    
    def _handle_union_annotation_with_instance(self, union_annotation_arg, _instance_arg, _typevar_value_groups):
        """Handle union type in annotation with corresponding instance data."""
        
        # This method is currently unused as the recursive approach handles union types
        # more effectively. Keeping it as a placeholder for potential future use.
        # The actual union handling is now done in _handle_union_value_binding.
        
        # Get the union TypeVars
        union_args_info = get_concrete_args(union_annotation_arg.resolved_type)
        union_typevars = []
        for arg_info in union_args_info:
            if isinstance(arg_info.origin, TypeVar):
                union_typevars.append(arg_info.origin)
        
        if not union_typevars:
            return  # No TypeVars in the union
        
        # Current implementation relies on the recursive processing in 
        # _extract_typevars_from_generic_info to handle union types properly
    
    def _fallback_instance_type_extraction(self, _annotation: Any, value: Any, source: str):
        """
        Fallback mechanism to extract type information from instances when annotation lacks TypeVars.
        
        This handles cases where Python's signature processing loses TypeVar information
        (e.g., List[Box[A]] becomes List[Box]) but the instance still contains concrete types.
        """
        
        try:
            # Extract concrete types found in the instance
            found_concrete_types = self._extract_concrete_types_from_instance(value)
            
            if found_concrete_types:
                # We found concrete types in the instance (e.g., int from Box[int])
                # but the annotation didn't have TypeVars to bind them to.
                # 
                # Strategy: Create a "fallback constraint" that can be used if there are
                # unbound TypeVars elsewhere in the function signature (like in the return type)
                
                # For simplicity, let's assume that if we find a single concrete type,
                # it can be bound to any unbound TypeVar that needs a binding.
                # This is a heuristic that works for many common cases.
                
                if len(found_concrete_types) == 1:
                    concrete_type = next(iter(found_concrete_types))
                    
                    # Create a special constraint that can bind any unbound TypeVar to this type
                    # We'll use a special marker to indicate this is a fallback constraint
                    self._create_fallback_constraint(concrete_type, source)
                
        except (TypeError, AttributeError, ValueError):
            # If extraction fails, just skip the fallback
            pass
    
    def _extract_concrete_types_from_instance(self, instance: Any) -> Set[type]:
        """Extract concrete types from an instance, looking into containers."""
        found_types = set()
        
        if isinstance(instance, (list, tuple)):
            for item in instance:
                item_info = self.generic_utils.get_instance_generic_info(item)
                if item_info.concrete_args:
                    # This item has concrete type arguments (like Box[int])
                    for arg_info in item_info.concrete_args:
                        found_types.add(arg_info.resolved_type)
                        
        elif isinstance(instance, dict):
            for key, val in instance.items():
                key_info = self.generic_utils.get_instance_generic_info(key)
                val_info = self.generic_utils.get_instance_generic_info(val)
                if key_info.concrete_args:
                    for arg_info in key_info.concrete_args:
                        found_types.add(arg_info.resolved_type)
                if val_info.concrete_args:
                    for arg_info in val_info.concrete_args:
                        found_types.add(arg_info.resolved_type)
        else:
            # Single instance
            instance_info = self.generic_utils.get_instance_generic_info(instance)
            if instance_info.concrete_args:
                for arg_info in instance_info.concrete_args:
                    found_types.add(arg_info.resolved_type)
        
        return found_types
    
    def _extract_concrete_types_from_value(self, value: Any) -> List[type]:
        """
        Extract concrete types from a value, intelligently handling nested structures.
        
        This method drills down into complex values to find the actual concrete types
        that should be bound to TypeVars, rather than just returning the top-level type.
        """
        concrete_types = []
        
        if isinstance(value, (list, tuple, set)):
            # For containers, extract types from each element
            for item in value:
                # First check if the item itself has generic type information
                item_info = self.generic_utils.get_instance_generic_info(item)
                if item_info.concrete_args:
                    # Extract types from the concrete args (e.g., Box[int] -> int)
                    for arg_info in item_info.concrete_args:
                        self._collect_concrete_types_recursive(arg_info.resolved_type, concrete_types)
                else:
                    # No generic info, use the item's type directly
                    concrete_types.append(_infer_type_from_value(item))
        
        elif isinstance(value, dict):
            # For dicts, we might need to extract from both keys and values
            # but this depends on the context - for now, try to extract from values
            # as they're more likely to contain the generic type information we need
            for val in value.values():
                val_info = self.generic_utils.get_instance_generic_info(val)
                if val_info.concrete_args:
                    for arg_info in val_info.concrete_args:
                        self._collect_concrete_types_recursive(arg_info.resolved_type, concrete_types)
                else:
                    # Check if the value is itself a container we can drill into
                    if isinstance(val, (list, tuple, set, dict)):
                        concrete_types.extend(self._extract_concrete_types_from_value(val))
                    else:
                        concrete_types.append(_infer_type_from_value(val))
        
        else:
            # Single value - check if it has generic type info first
            value_info = self.generic_utils.get_instance_generic_info(value)
            if value_info.concrete_args:
                # Extract from concrete args (e.g., Wrap[float] -> float)
                for arg_info in value_info.concrete_args:
                    self._collect_concrete_types_recursive(arg_info.resolved_type, concrete_types)
            else:
                # Simple value, use its type - but be conservative about type objects
                value_type = _infer_type_from_value(value)
                # Don't extract meta-types like 'type' unless it's clearly intended
                if value_type not in (type, bool, str) or isinstance(value, (bool, str)):
                    concrete_types.append(value_type)
        
        return concrete_types
    
    def _collect_concrete_types_recursive(self, resolved_type: Any, concrete_types: List[type]):
        """Recursively collect concrete types from a resolved type."""
        # Check if this is a basic type
        if isinstance(resolved_type, type) and not hasattr(resolved_type, '__origin__'):
            concrete_types.append(resolved_type)
            return
        
        # Check if this is a generic type with more args to extract
        origin = get_generic_origin(resolved_type)
        args_info = get_concrete_args(resolved_type)
        
        if origin and args_info:
            # This is a parameterized generic - extract from args
            for arg_info in args_info:
                self._collect_concrete_types_recursive(arg_info.resolved_type, concrete_types)
        else:
            # Fallback to the type itself
            if isinstance(resolved_type, type):
                concrete_types.append(resolved_type)
            else:
                # Try to get the type of the resolved_type
                concrete_types.append(type(resolved_type))
    
    def _create_fallback_constraint(self, concrete_type: type, source: str):
        """Create a fallback constraint that can be applied to unbound TypeVars."""
        # This is a simplified approach: we'll store the concrete type as "fallback evidence"
        # that can be used during solving if there are unbound TypeVars
        
        # For now, let's just store this as metadata that can be used during solving
        self._fallback_evidence.append((concrete_type, source))
            
    def _handle_annotation_fallback(self, annotation: Any, value: Any, source: str):
        """Fallback method for handling annotations that can't be processed by the unified approach."""
        
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
        
        # Handle case where TypeVars exist in type_params but not properly in concrete_args
        # This happens when complex nested annotations lose TypeVar information
        # Only apply this for single TypeVar cases to avoid conflicts
        if (len(ann_info.type_params) == 1 and 
            hasattr(value, '__orig_class__') and 
            not any(isinstance(arg_info.resolved_type, TypeVar) for arg_info in ann_info.concrete_args)):
            
            # Use the complete type information from the instance __orig_class__
            orig_class = value.__orig_class__
            orig_info = self.generic_utils.get_generic_info(orig_class)
            
            if orig_info.concrete_args:
                # Extract all concrete types from the complete instance type
                all_concrete_types = set()
                for arg_info in orig_info.concrete_args:
                    all_concrete_types.update(self._extract_concrete_types_from_arg(arg_info.resolved_type))
                
                # Bind the single TypeVar to the most basic concrete type found
                if all_concrete_types:
                    # Prefer basic types over complex ones
                    basic_types = {t for t in all_concrete_types if not hasattr(t, '__origin__')}
                    if basic_types:
                        concrete_type = next(iter(basic_types))  # Pick the first basic type
                        single_typevar = ann_info.type_params[0]
                        self.add_equality_constraint(single_typevar, concrete_type, f"{source}:instance_extraction", Variance.INVARIANT)
        
        # Create constraints for each type parameter (original logic)
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
            # For single concrete types inferred from homogeneous containers,
            # always create equality constraints regardless of variance.
            # Variance rules only apply when we have actual type conflicts.
            self.add_equality_constraint(typevar, inferred_type, source, variance)
        
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
        # Check if the value has richer type information than the annotation
        # This handles cases where TypeVars are lost in complex nested annotations
        # but preserved in the actual instances
        
        # Try to extract type information from the instance
        try:
            instance_info = self.generic_utils.get_instance_concrete_args(value)
            if instance_info:
                # The instance has generic type information that the annotation lacks
                # Try to extract TypeVars from a broader context if available
                if hasattr(value, '__orig_class__'):
                    # Use the original class information to find TypeVars
                    orig_class = value.__orig_class__
                    orig_info = self.generic_utils.get_generic_info(orig_class)
                    if orig_info.type_params:
                        # Map between annotation type and instance type to infer TypeVars
                        self._map_annotation_to_instance_types(annotation, orig_class, source)
                        return
        except (AttributeError, TypeError):
            pass
        
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
        
        # Note: We don't immediately fail on empty domains here because
        # they might be handled by the fallback mechanism
                
        # Generate solution
        solution = CSPSolution()
        unbound_typevars = []
        
        for typevar, domain in self.domains.items():
            try:
                best_type = domain.get_best_type()
                solution.bind(typevar, best_type)
            except CSPTypeInferenceError:
                # Domain is empty - this TypeVar couldn't be bound
                unbound_typevars.append(typevar)
        
        # Apply fallback evidence to unbound TypeVars
        if unbound_typevars and hasattr(self, '_fallback_evidence') and self._fallback_evidence:
            self._apply_fallback_evidence(unbound_typevars, solution)
            
        return solution
    
    def _apply_fallback_evidence(self, unbound_typevars: List[TypeVar], solution: CSPSolution):
        """Apply fallback evidence to unbound TypeVars."""
        
        # Simple strategy: if we have concrete types from fallback evidence,
        # and we have unbound TypeVars, try to bind them
        
        available_evidence = self._fallback_evidence.copy()
        
        for typevar in unbound_typevars:
            if available_evidence:
                # Use the first available concrete type
                concrete_type, source = available_evidence.pop(0)
                solution.bind(typevar, concrete_type)
                # Add a note about the fallback binding
                solution.conflicts.append(f"Fallback binding: {typevar} -> {concrete_type} from {source}")
        
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
    
    def _map_annotation_to_instance_types(self, _annotation_type: Any, instance_type: Any, source: str):
        """Map between incomplete annotation and complete instance type to infer TypeVars.
        
        This handles cases where the annotation loses TypeVar information (like Wrap[List[Box]])
        but the instance preserves it (like Wrap[List[Box[int]]]).
        """
        
        # Get type information from the complete instance type
        instance_info = self.generic_utils.get_generic_info(instance_type)
        
        if not instance_info.type_params:
            return  # No TypeVars to infer
        
        # The key insight: extract TypeVars from the complete instance type
        # and create equality constraints for them based on the concrete types
        # found in the instance structure
        
        for typevar in instance_info.type_params:
            # For each TypeVar in the instance type, try to find its concrete binding
            # by looking at the concrete args
            if instance_info.concrete_args:
                # Map TypeVar to its position and extract the corresponding concrete type
                # This is a simplified approach - in practice we'd need more sophisticated mapping
                concrete_types = set()
                for arg_info in instance_info.concrete_args:
                    # Recursively extract concrete types from the arg
                    concrete_types.update(self._extract_concrete_types_from_arg(arg_info.resolved_type))
                
                if concrete_types:
                    if len(concrete_types) == 1:
                        concrete_type = next(iter(concrete_types))
                        self.add_equality_constraint(typevar, concrete_type, f"{source}:instance_inferred")
                    else:
                        # Multiple concrete types found - create subset constraint
                        self.add_subset_constraint({typevar}, concrete_types, Variance.COVARIANT, f"{source}:instance_inferred")
    
    def _extract_concrete_types_from_arg(self, arg_type: Any) -> Set[type]:
        """Extract concrete types from a type argument, handling nested generics."""
        concrete_types = set()
        
        # First check if this is a generic type with concrete args
        origin = get_generic_origin(arg_type)
        args_info = get_concrete_args(arg_type)
        
        if origin and args_info:
            # This is a parameterized generic type (like Box[int])
            # Extract from the concrete arguments recursively
            for arg_info in args_info:
                concrete_types.update(self._extract_concrete_types_from_arg(arg_info.resolved_type))
        elif isinstance(arg_type, type):
            # This is a basic type (like int, str, etc.)
            concrete_types.add(arg_type)
        else:
            # Fallback for other cases
            concrete_types.add(arg_type)
        
        return concrete_types
    
    def _find_typevar_binding_in_args(self, annotation_arg: Any, instance_arg: Any, target_typevar: TypeVar) -> Set[type]:
        """Find concrete types that bind to a TypeVar by comparing annotation and instance arguments.
        
        For example: annotation_arg = list[Box], instance_arg = list[Box[int]], target_typevar = A
        Should extract that A = int.
        """
        concrete_types = set()
        
        # Get origins to compare structure
        ann_origin = get_generic_origin(annotation_arg)
        inst_origin = get_generic_origin(instance_arg)
        
        # If origins match, compare their arguments
        if ann_origin and inst_origin and ann_origin == inst_origin:
            ann_args = get_concrete_args(annotation_arg)
            inst_args = get_concrete_args(instance_arg)
            
            # Recursively compare arguments
            for ann_arg_info, inst_arg_info in zip(ann_args, inst_args):
                ann_sub_arg = ann_arg_info.resolved_type
                inst_sub_arg = inst_arg_info.resolved_type
                
                # Recursively find bindings in nested arguments
                concrete_types.update(self._find_typevar_binding_in_args(ann_sub_arg, inst_sub_arg, target_typevar))
        
        # Check if annotation and instance have same origin but instance has concrete args
        # This handles: annotation_arg = Box (no args), instance_arg = Box[int] (with args)
        elif ann_origin and inst_origin and ann_origin == inst_origin:
            ann_args = get_concrete_args(annotation_arg)
            inst_args = get_concrete_args(instance_arg)
            
            # If annotation has no concrete args but instance does, extract from instance
            if not ann_args and inst_args:
                for inst_arg_info in inst_args:
                    concrete_types.update(self._extract_concrete_types_from_arg(inst_arg_info.resolved_type))
        
        # Fallback: check by class name if origins don't match
        elif hasattr(annotation_arg, '__name__') and hasattr(instance_arg, '__name__') and annotation_arg.__name__ == instance_arg.__name__:
            # Same class name - try to extract args from instance
            inst_args = get_concrete_args(instance_arg)
            for inst_arg_info in inst_args:
                concrete_types.update(self._extract_concrete_types_from_arg(inst_arg_info.resolved_type))
        
        return concrete_types


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
        
    # Before solving, ensure domains exist for TypeVars in the return annotation
    # This handles cases where TypeVars appear in return annotation but not parameters
    return_typevars = _find_unbound_typevars(return_annotation)
    for typevar in return_typevars:
        if typevar not in engine.domains:
            engine.domains[typevar] = TypeDomain(typevar)
    
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


def _types_are_compatible(type1: type, type2: type) -> bool:
    """Check if two types are compatible (have inheritance relationship)."""
    try:
        return issubclass(type1, type2) or issubclass(type2, type1)
    except TypeError:
        # Handle cases where types aren't classes
        return type1 == type2
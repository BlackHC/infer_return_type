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


@dataclass
class TypeConstraint:
    """A single constraint in our CSP model."""
    constraint_type: ConstraintType
    variables: Set[TypeVar]  # TypeVars involved in this constraint
    types: Set[type]        # Concrete types involved
    description: str        # Human-readable description
    priority: int = 1       # Higher priority constraints are satisfied first
    source: str = ""        # Where this constraint came from (for debugging)
    
    def __str__(self):
        vars_str = ", ".join(str(v) for v in self.variables)
        types_str = ", ".join(str(t) for t in self.types)
        return f"{self.constraint_type.value}: vars={{{vars_str}}} types={{{types_str}}} - {self.description}"


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
                
    def add_equality_constraint(self, typevar: TypeVar, concrete_type: type, source: str = ""):
        """Add A = type constraint."""
        constraint = TypeConstraint(
            constraint_type=ConstraintType.EQUALITY,
            variables={typevar},
            types={concrete_type},
            description=f"{typevar} = {concrete_type}",
            priority=10,  # High priority - exact constraints
            source=source
        )
        self.add_constraint(constraint)
        
    def add_subset_constraint(self, typevars: Set[TypeVar], concrete_types: Set[type], source: str = ""):
        """Add {TypeVars} ⊇ {concrete_types} constraint (union constraint)."""
        constraint = TypeConstraint(
            constraint_type=ConstraintType.SUBSET,
            variables=typevars,
            types=concrete_types,
            description=f"{{{', '.join(str(v) for v in typevars)}}} ⊇ {{{', '.join(str(t) for t in concrete_types)}}}",
            priority=5,
            source=source
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
            source=source
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
            self.add_equality_constraint(annotation, concrete_type, f"{source}:direct")
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
            
        # Handle generic containers
        if origin in (list, List):
            self._handle_list_annotation(annotation, value, source)
        elif origin in (dict, Dict):
            self._handle_dict_annotation(annotation, value, source)
        elif origin in (tuple, Tuple):
            self._handle_tuple_annotation(annotation, value, source)
        elif origin in (set, Set):
            self._handle_set_annotation(annotation, value, source)
        else:
            # Handle other generic types using unified interface
            self._handle_custom_generic_annotation(annotation, value, source)
            
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
                self.add_equality_constraint(best_match, concrete_type, f"{source}:union_direct")
                self.add_bounds_constraint(best_match, f"{source}:union_bounds")
            else:
                # Process the matching alternative
                self._collect_constraints_recursive(best_match, value, f"{source}:union_match")
        else:
            # Fallback: if we have TypeVars and no good match, create subset constraint
            if typevars:
                self.add_subset_constraint(typevars, {value_type}, f"{source}:union_fallback")
            
    def _handle_list_annotation(self, annotation: Any, value: Any, source: str):
        """Handle List[T] annotations.""" 
        if not isinstance(value, list):
            raise CSPTypeInferenceError(f"Expected list, got {type(value)} in {source}")
            
        args = get_concrete_args(annotation)
        if len(args) == 1:
            element_annotation = args[0]
            
            if isinstance(element_annotation, TypeVar):
                # Covariant constraint: TypeVar can be union of all element types
                if value:
                    element_types = {type(item) for item in value}
                    if len(element_types) == 1:
                        # Homogeneous list - equality constraint
                        self.add_equality_constraint(element_annotation, next(iter(element_types)), f"{source}:list_homogeneous")
                    else:
                        # Mixed list - subset constraint (union formation)
                        self.add_subset_constraint({element_annotation}, element_types, f"{source}:list_mixed")
                    self.add_bounds_constraint(element_annotation, f"{source}:list_bounds")
            else:
                # Nested generic - recurse into each element
                for i, item in enumerate(value):
                    self._collect_constraints_recursive(element_annotation, item, f"{source}:list[{i}]")
                    
    def _handle_dict_annotation(self, annotation: Any, value: Any, source: str):
        """Handle Dict[K, V] annotations."""
        if not isinstance(value, dict):
            raise CSPTypeInferenceError(f"Expected dict, got {type(value)} in {source}")
            
        args = get_concrete_args(annotation)
        if len(args) == 2:
            key_annotation, value_annotation = args
            
            if isinstance(key_annotation, TypeVar) and value:
                key_types = {type(k) for k in value.keys()}
                if len(key_types) == 1:
                    self.add_equality_constraint(key_annotation, next(iter(key_types)), f"{source}:dict_keys")
                else:
                    self.add_subset_constraint({key_annotation}, key_types, f"{source}:dict_keys_mixed")
                self.add_bounds_constraint(key_annotation, f"{source}:dict_key_bounds")
                
            if isinstance(value_annotation, TypeVar) and value:
                value_types = {type(v) for v in value.values()}
                if len(value_types) == 1:
                    self.add_equality_constraint(value_annotation, next(iter(value_types)), f"{source}:dict_values")
                else:
                    self.add_subset_constraint({value_annotation}, value_types, f"{source}:dict_values_mixed")
                self.add_bounds_constraint(value_annotation, f"{source}:dict_value_bounds")
                
            # Handle non-TypeVar annotations recursively
            if not isinstance(key_annotation, TypeVar):
                for key in value.keys():
                    self._collect_constraints_recursive(key_annotation, key, f"{source}:dict_key")
                    
            if not isinstance(value_annotation, TypeVar):
                for val in value.values():
                    self._collect_constraints_recursive(value_annotation, val, f"{source}:dict_value")
                    
    def _handle_set_annotation(self, annotation: Any, value: Any, source: str):
        """Handle Set[T] annotations."""
        if not isinstance(value, set):
            raise CSPTypeInferenceError(f"Expected set, got {type(value)} in {source}")
            
        args = get_concrete_args(annotation)
        if len(args) == 1:
            element_annotation = args[0]
            
            if isinstance(element_annotation, TypeVar):
                if value:
                    element_types = {type(item) for item in value}
                    if len(element_types) == 1:
                        self.add_equality_constraint(element_annotation, next(iter(element_types)), f"{source}:set_homogeneous")
                    else:
                        self.add_subset_constraint({element_annotation}, element_types, f"{source}:set_mixed")
                    self.add_bounds_constraint(element_annotation, f"{source}:set_bounds")
            else:
                # Handle Union inside Set specially
                origin = get_generic_origin(element_annotation)
                if origin is Union:
                    union_args = get_concrete_args(element_annotation)
                    typevars_in_union = {arg for arg in union_args if isinstance(arg, TypeVar)}
                    if typevars_in_union:
                        # Set[Union[A, B]] with {int, str} means {A, B} ⊇ {int, str}
                        element_types = {type(item) for item in value}
                        self.add_subset_constraint(typevars_in_union, element_types, f"{source}:set_union")
                else:
                    # Recurse into each element
                    for i, item in enumerate(value):
                        self._collect_constraints_recursive(element_annotation, item, f"{source}:set[{i}]")
                        
    def _handle_tuple_annotation(self, annotation: Any, value: Any, source: str):
        """Handle Tuple annotations."""
        if not isinstance(value, tuple):
            raise CSPTypeInferenceError(f"Expected tuple, got {type(value)} in {source}")
            
        args = get_concrete_args(annotation)
        
        if len(args) == 2 and args[1] is ...:
            # Variable length tuple: Tuple[T, ...]
            element_annotation = args[0]
            if isinstance(element_annotation, TypeVar) and value:
                element_types = {type(item) for item in value}
                if len(element_types) == 1:
                    self.add_equality_constraint(element_annotation, next(iter(element_types)), f"{source}:var_tuple")
                else:
                    self.add_subset_constraint({element_annotation}, element_types, f"{source}:var_tuple_mixed")
                self.add_bounds_constraint(element_annotation, f"{source}:var_tuple_bounds")
        else:
            # Fixed length tuple
            for i, item in enumerate(value):
                if i < len(args):
                    self._collect_constraints_recursive(args[i], item, f"{source}:tuple[{i}]")
                    
    def _handle_custom_generic_annotation(self, annotation: Any, value: Any, source: str):
        """Handle custom generic annotations using unified generic_utils interface."""
        
        # Use generic_utils to extract information consistently
        ann_info = self.generic_utils.get_generic_info(annotation)
        val_info = self.generic_utils.get_instance_generic_info(value)
        
        # If both annotation and value are recognized as generic
        if ann_info.is_generic and val_info.is_generic:
            # Check if they have compatible origins
            if ann_info.origin == val_info.origin or (
                hasattr(ann_info.origin, '__name__') and hasattr(val_info.origin, '__name__') and
                ann_info.origin.__name__ == val_info.origin.__name__
            ):
                # Try to align type parameters with concrete arguments
                if len(ann_info.type_params) == len(val_info.concrete_args):
                    for param, concrete in zip(ann_info.type_params, val_info.concrete_args):
                        # Use equality constraint for custom types as default
                        self.add_equality_constraint(param, concrete, f"{source}:custom_generic")
                        self.add_bounds_constraint(param, f"{source}:custom_bounds")
                    return
        
        # If annotation has type parameters but no concrete args from value,
        # try to extract from nested structure
        if ann_info.type_params:
            self._extract_from_nested_structure(annotation, value, source)
            return
        
        # Fallback: try to extract from nested structure
        self._extract_from_nested_structure(annotation, value, source)
    
    def _extract_from_nested_structure(self, annotation: Any, value: Any, source: str):
        """Fallback method to extract constraints from nested structures."""
        
        # Try to align annotation structure with value structure
        ann_info = self.generic_utils.get_generic_info(annotation)
        
        # First, check if the value has explicit type information
        if hasattr(value, '__orig_class__'):
            value_type = value.__orig_class__
            val_info = self.generic_utils.get_generic_info(value_type)
            
            # Recursively align the concrete_args structures
            if ann_info.concrete_args and val_info.concrete_args and len(ann_info.concrete_args) == len(val_info.concrete_args):
                for ann_arg, val_arg in zip(ann_info.concrete_args, val_info.concrete_args):
                    # Recursively handle nested generic structures
                    self._extract_from_annotation_alignment(ann_arg, val_arg, source)
                return
        
        # Try to extract from instance fields for dataclasses and similar
        self._extract_from_instance_fields(annotation, value, source)
    
    def _extract_from_annotation_alignment(self, ann_type: Any, val_type: Any, source: str):
        """Extract constraints by aligning annotation and value type structures."""
        
        ann_info = self.generic_utils.get_generic_info(ann_type)
        val_info = self.generic_utils.get_generic_info(val_type)
        
        # Special handling for Pydantic and other generic classes
        if (not ann_info.concrete_args and ann_info.type_params and 
            val_info.concrete_args and len(ann_info.type_params) == len(val_info.concrete_args)):
            
            # Get TypeVars from annotation and concrete types from value type
            for ann_typevar, val_concrete in zip(ann_info.type_params, val_info.concrete_args):
                if isinstance(ann_typevar, TypeVar):
                    self.add_equality_constraint(ann_typevar, val_concrete, f"{source}:alignment")
                    self.add_bounds_constraint(ann_typevar, f"{source}:alignment_bounds")
            return
        
        # If both have the same origin and compatible args, align them
        if (ann_info.origin == val_info.origin and ann_info.concrete_args and val_info.concrete_args and 
            len(ann_info.concrete_args) == len(val_info.concrete_args)):
            for i, (ann_arg, val_arg) in enumerate(zip(ann_info.concrete_args, val_info.concrete_args)):
                if isinstance(ann_arg, TypeVar):
                    self.add_equality_constraint(ann_arg, val_arg, f"{source}:alignment_arg_{i}")
                    self.add_bounds_constraint(ann_arg, f"{source}:alignment_arg_{i}_bounds")
                else:
                    # Recursively handle deeper nesting
                    self._extract_from_annotation_alignment(ann_arg, val_arg, f"{source}:nested_{i}")
        elif isinstance(ann_type, TypeVar):
            # Direct TypeVar binding
            self.add_equality_constraint(ann_type, val_type, f"{source}:direct_alignment")
            self.add_bounds_constraint(ann_type, f"{source}:direct_alignment_bounds")
    
    def _extract_from_instance_fields(self, annotation: Any, value: Any, source: str):
        """Extract type constraints from instance field values."""
        
        # Use generic_utils for consistent field extraction
        ann_info = self.generic_utils.get_generic_info(annotation)
        
        # Handle dataclasses
        from dataclasses import is_dataclass, fields
        if is_dataclass(value):
            if ann_info.type_params:
                # Try to infer TypeVar bindings from field values
                field_values = []
                for field in fields(value):
                    field_value = getattr(value, field.name)
                    field_values.append((field.name, field_value, type(field_value)))
                
                # For now, simple heuristic: if there's one TypeVar and all fields have same type
                if len(ann_info.type_params) == 1:
                    if field_values:
                        # Collect types from all fields and create union
                        field_types = {fv[2] for fv in field_values}
                        if len(field_types) == 1:
                            self.add_equality_constraint(ann_info.type_params[0], next(iter(field_types)), f"{source}:dataclass_field")
                        else:
                            self.add_subset_constraint({ann_info.type_params[0]}, field_types, f"{source}:dataclass_fields_mixed")
                        self.add_bounds_constraint(ann_info.type_params[0], f"{source}:dataclass_bounds")
        
        # Handle other generic instances by trying to extract from __dict__
        elif hasattr(value, '__dict__'):
            if ann_info.type_params and len(ann_info.type_params) == 1:
                # Extract types from instance attributes
                attr_types = {type(v) for v in value.__dict__.values() if v is not None}
                if attr_types:
                    if len(attr_types) == 1:
                        self.add_equality_constraint(ann_info.type_params[0], next(iter(attr_types)), f"{source}:instance_attr")
                    else:
                        self.add_subset_constraint({ann_info.type_params[0]}, attr_types, f"{source}:instance_attrs_mixed")
                    self.add_bounds_constraint(ann_info.type_params[0], f"{source}:instance_bounds")
    
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
            self.domains[typevar].set_exact_type(concrete_type)
            
        elif constraint.constraint_type == ConstraintType.SUBSET:
            # {TypeVars} ⊇ {concrete_types} - distribute types among TypeVars
            if len(constraint.variables) == 1:
                # Single TypeVar must be union of all types
                typevar = next(iter(constraint.variables))
                if len(constraint.types) == 1:
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
            # A ≤ SuperType
            typevar = next(iter(constraint.variables))
            supertype = next(iter(constraint.types))
            self.domains[typevar].add_subtype_constraint(supertype)
            
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
        engine.add_equality_constraint(typevar, override_type, "override")
        
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
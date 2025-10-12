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
from dataclasses import fields, is_dataclass
from enum import Enum
from collections import defaultdict

# Import unified generic utilities
from generic_utils import (
    GenericTypeUtils, get_generic_info, get_concrete_args,
    get_generic_origin, create_union_if_needed, get_annotation_value_pairs
)


class UnificationError(Exception):
    """Raised when unification fails."""
    pass


class TypeInferenceError(Exception):
    """Raised when type inference fails."""
    pass


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
        # Use the unified generic type utils instead of custom extractors
        self.generic_utils = GenericTypeUtils()
    
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
        
        # Handle Union types using generic_utils
        origin = get_generic_origin(annotation)
        args = get_concrete_args(annotation)
        
        if origin is Union or (hasattr(types, 'UnionType') and origin is getattr(types, 'UnionType')):
            self._handle_union_constraints(annotation, value, constraints)
            return
        
        # Handle Optional (Union[T, None])
        if origin is Union and len(args) == 2:
            # Check if one of the args is NoneType
            has_none = any(arg_info.origin is type(None) for arg_info in args)
            if has_none:
                if value is not None:
                    # Find the non-None type
                    non_none_type = args[0].resolved_type if args[1].origin is type(None) else args[1].resolved_type
                    self._collect_constraints_internal(non_none_type, value, constraints)
                else:
                    # Value is None - we still need to handle the TypeVar
                    non_none_type_info = args[0] if args[1].origin is type(None) else args[1]
                    if isinstance(non_none_type_info.origin, TypeVar):
                        # For Optional[A] with None value, we can't infer A but shouldn't fail
                        pass
                return
        
        # Handle generic containers
        if origin in (list, List):
            self._handle_list_constraints(annotation, value, constraints)
        elif origin in (dict, Dict):
            self._handle_dict_constraints(annotation, value, constraints)
        elif origin in (tuple, Tuple):
            self._handle_tuple_constraints(annotation, value, constraints)
        elif origin in (set, Set):
            self._handle_set_constraints(annotation, value, constraints)
        else:
            # Handle custom generic types using unified interface
            self._handle_custom_generic_constraints(annotation, value, constraints)
    
    def _add_covariant_constraints_for_elements(
        self, typevar: TypeVar, values, constraints: List[Constraint]
    ):
        """
        Add separate covariant constraints for each distinct type in values.
        
        This allows proper union formation and bound checking in the constraint solver.
        """
        element_types = {type(item) for item in values}
        for element_type in element_types:
            constraints.append(Constraint(typevar, element_type, Variance.COVARIANT))
    
    def _handle_union_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Union type constraints by trying each alternative."""
        args = get_concrete_args(annotation)
        
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
                    alt_origin = get_generic_origin(alternative_info.resolved_type)
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
    
    def _handle_list_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle List[T] constraints."""
        if not isinstance(value, list):
            raise UnificationError(f"Expected list, got {type(value)}")
        
        args = get_concrete_args(annotation)
        if len(args) == 1:
            element_annotation_info = args[0]
            
            if isinstance(element_annotation_info.origin, TypeVar):
                # Collect types from all elements
                if value:
                    # Create separate constraints for each distinct type
                    # This allows proper union formation and bound checking
                    self._add_covariant_constraints_for_elements(
                        element_annotation_info.origin, value, constraints
                    )
                # If empty list, don't add constraint - will be handled later
            else:
                # Handle Union inside List specially
                element_annotation = element_annotation_info.resolved_type
                origin = get_generic_origin(element_annotation)
                if origin is Union:
                    # For List[Union[A, B]] with values [int, str], we need to collect constraints differently
                    union_args = get_concrete_args(element_annotation)
                    
                    # Try to distribute types among TypeVars in the union
                    if self._try_distribute_union_types(set(value), union_args, constraints):
                        return
                    
                    # Fallback: match each item to union alternatives
                    for item in value:
                        self._match_value_to_union_alternatives(item, union_args, constraints)
                else:
                    # Recursively handle each element
                    for item in value:
                        self._collect_constraints_internal(element_annotation, item, constraints)
    
    def _handle_dict_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Dict[K, V] constraints."""
        if not isinstance(value, dict):
            raise UnificationError(f"Expected dict, got {type(value)}")
        
        args = get_concrete_args(annotation)
        if len(args) == 2:
            key_annotation_info, value_annotation_info = args
            
            if isinstance(key_annotation_info.origin, TypeVar) and value:
                key_types = {type(k) for k in value.keys()}
                union_type = create_union_if_needed(key_types)
                constraints.append(Constraint(key_annotation_info.origin, union_type, Variance.INVARIANT))
            
            if isinstance(value_annotation_info.origin, TypeVar) and value:
                value_types = {type(v) for v in value.values()}
                union_type = create_union_if_needed(value_types)
                constraints.append(Constraint(value_annotation_info.origin, union_type, Variance.COVARIANT))
            
            # Recursively handle non-TypeVar annotations
            if not isinstance(key_annotation_info.origin, TypeVar):
                key_annotation = key_annotation_info.resolved_type
                for key in value.keys():
                    self._collect_constraints_internal(key_annotation, key, constraints)
            
            if not isinstance(value_annotation_info.origin, TypeVar):
                value_annotation = value_annotation_info.resolved_type
                for val in value.values():
                    self._collect_constraints_internal(value_annotation, val, constraints)
    
    def _handle_tuple_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Tuple constraints - process sub-constraints in order for better context."""
        if not isinstance(value, tuple):
            raise UnificationError(f"Expected tuple, got {type(value)}")
        
        args = get_concrete_args(annotation)
        
        if len(args) == 2 and args[1].origin is ...:
            # Variable length tuple: Tuple[T, ...]
            element_annotation_info = args[0]
            if isinstance(element_annotation_info.origin, TypeVar) and value:
                # Create separate constraints for each distinct type
                # This allows proper union formation and bound checking
                self._add_covariant_constraints_for_elements(
                    element_annotation_info.origin, value, constraints
                )
            elif not isinstance(element_annotation_info.origin, TypeVar):
                element_annotation = element_annotation_info.resolved_type
                for item in value:
                    self._collect_constraints_internal(element_annotation, item, constraints)
        else:
            # Fixed length tuple: Process in order for better constraint propagation
            for i, item in enumerate(value):
                if i < len(args):
                    arg_info = args[i]
                    if isinstance(arg_info.origin, TypeVar):
                        # Direct TypeVar constraint
                        item_type = type(item)
                        constraints.append(Constraint(arg_info.origin, item_type, Variance.INVARIANT))
                    else:
                        # Process each tuple element, building up constraint context
                        self._collect_constraints_internal(arg_info.resolved_type, item, constraints)
    
    def _handle_set_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle Set[T] constraints."""
        if not isinstance(value, set):
            raise UnificationError(f"Expected set, got {type(value)}")
        
        args = get_concrete_args(annotation)
        if len(args) == 1:
            element_annotation_info = args[0]
            
            if isinstance(element_annotation_info.origin, TypeVar):
                if value:
                    # Create separate constraints for each distinct type
                    # This allows proper union formation and bound checking
                    self._add_covariant_constraints_for_elements(
                        element_annotation_info.origin, value, constraints
                    )
            else:
                # Check if this is a Union type
                element_annotation = element_annotation_info.resolved_type
                origin = get_generic_origin(element_annotation)
                if origin is Union:
                    union_args = get_concrete_args(element_annotation)
                    
                    # Try to distribute types among TypeVars in the union
                    if self._try_distribute_union_types(value, union_args, constraints):
                        return
                    
                    # Fallback: match each item to union alternatives
                    for item in value:
                        self._match_value_to_union_alternatives(item, union_args, constraints)
                else:
                    for item in value:
                        self._collect_constraints_internal(element_annotation, item, constraints)
    
    def _handle_custom_generic_constraints(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Handle custom generic types using unified generic_utils interface.
        
        Tries multiple strategies in order of preference:
        1. Annotation-value pair extraction (most precise)
        2. Direct generic structure matching
        3. Nested structure extraction (fallback)
        """
        # Strategy 1: Try annotation-value pairs approach (works for dataclasses, Pydantic models)
        if self._try_extract_with_annotation_value_pairs(annotation, value, constraints):
            return
        
        # Strategy 2: Try direct generic structure matching (Generic[A, B] with Generic[int, str])
        ann_info = self.generic_utils.get_generic_info(annotation)
        val_info = self.generic_utils.get_instance_generic_info(value)
        
        if ann_info.is_generic and val_info.is_generic:
            if self._origins_are_compatible(ann_info.origin, val_info.origin):
                if len(ann_info.type_params) == len(val_info.concrete_args):
                    for param, concrete in zip(ann_info.type_params, val_info.concrete_args):
                        constraints.append(Constraint(param, concrete, Variance.INVARIANT))
                    return
        
        # Strategy 3: Fallback to nested structure extraction
        if ann_info.type_params:
            self._extract_from_nested_structure(annotation, value, constraints)
    
    def _origins_are_compatible(self, origin1: Any, origin2: Any) -> bool:
        """Check if two generic origins are compatible.
        
        Uses direct equality comparison, which works correctly for all types
        including classes, typing constructs, and generic origins.
        """
        return origin1 == origin2
    
    def _try_extract_with_annotation_value_pairs(self, annotation: Any, value: Any, constraints: List[Constraint]) -> bool:
        """Try to extract constraints using get_annotation_value_pairs with recursive drilling."""
        
        # First try direct generic structure matching for cases like PydanticModel[A, list[B]] with PydanticModel[int, list[str]]
        if self._try_direct_generic_structure_matching(annotation, value, constraints):
            return True
        
        try:
            pairs = get_annotation_value_pairs(annotation, value)
            if not pairs:
                return False
            
            for generic_info, concrete_value in pairs:
                self._process_generic_value_pair(generic_info, concrete_value, constraints)
            
            return True
        except Exception:
            # If anything goes wrong, fall back to original approach
            return False
    
    def _try_direct_generic_structure_matching(self, annotation: Any, value: Any, constraints: List[Constraint]) -> bool:
        """Try to match generic structures directly (e.g., Generic[A, B] with Generic[int, str])."""
        
        ann_info = self.generic_utils.get_generic_info(annotation)
        val_info = self.generic_utils.get_instance_generic_info(value)
        
        # Both must be generic with compatible origins and matching argument counts
        if not (ann_info.is_generic and val_info.is_generic):
            return False
            
        if not self._origins_are_compatible(ann_info.origin, val_info.origin):
            return False
        
        if len(ann_info.concrete_args) != len(val_info.concrete_args):
            return False
        
        # Match each type argument pair
        found_constraints = False
        for ann_arg, val_arg in zip(ann_info.concrete_args, val_info.concrete_args):
            if self._match_generic_arg_pair(ann_arg, val_arg, constraints):
                found_constraints = True
        
        return found_constraints
    
    def _match_generic_arg_pair(self, ann_arg: Any, val_arg: Any, constraints: List[Constraint]) -> bool:
        """Match a pair of generic arguments and extract constraints."""
        
        # Case 1: annotation has TypeVar, value has concrete type
        if isinstance(ann_arg.origin, TypeVar) and not isinstance(val_arg.origin, TypeVar):
            constraints.append(Constraint(ann_arg.origin, val_arg.resolved_type, Variance.INVARIANT))
            return True
        
        # Case 2: both have the same structure - recursively match
        elif (ann_arg.origin == val_arg.origin and 
              len(ann_arg.concrete_args) == len(val_arg.concrete_args) and
              ann_arg.concrete_args and val_arg.concrete_args):
            found_any = False
            for sub_ann, sub_val in zip(ann_arg.concrete_args, val_arg.concrete_args):
                if self._match_generic_arg_pair(sub_ann, sub_val, constraints):
                    found_any = True
            return found_any
        
        # Case 3: annotation has nested TypeVar, value has concrete structure
        elif ann_arg.concrete_args and isinstance(ann_arg.concrete_args[0].origin, TypeVar):
            # For example: list[B] matches list[str], extract B -> str
            if (ann_arg.origin == val_arg.origin and val_arg.concrete_args and 
                not isinstance(val_arg.concrete_args[0].origin, TypeVar)):
                typevar = ann_arg.concrete_args[0].origin
                concrete_type = val_arg.concrete_args[0].resolved_type
                constraints.append(Constraint(typevar, concrete_type, Variance.COVARIANT))
                return True
        
        return False
    
    def _process_generic_value_pair(self, generic_info: Any, concrete_value: Any, constraints: List[Constraint]):
        """Process a (GenericInfo, value) pair and extract constraints, with recursive drilling."""
        
        if isinstance(generic_info.origin, TypeVar):
            # Direct TypeVar - we need to find what this TypeVar should actually represent
            # by looking at the context of the annotation structure
            self._resolve_typevar_from_context(generic_info, concrete_value, constraints)
        else:
            # Complex GenericInfo - try to recursively extract
            self._collect_constraints_internal(generic_info.resolved_type, concrete_value, constraints)
    
    def _resolve_typevar_from_context(self, generic_info: Any, concrete_value: Any, constraints: List[Constraint]):
        """Resolve TypeVar constraints by finding what the TypeVar should represent in context."""
        
        typevar = generic_info.origin
        
        # The key insight: generic_info represents a TypeVar that was extracted from some position
        # in the annotation structure. We need to find what annotation structure should be used
        # to process the concrete_value.
        
        # For now, let's try a different approach: 
        # If we have a TypeVar and a complex value (like a list of generic objects),
        # try to infer the TypeVar from the elements of the value
        
        if isinstance(concrete_value, list) and concrete_value:
            # We have A -> [Box[int](...), Box[int](...)]
            # Let's extract the types from the Box[int] elements
            element_types = set()
            for element in concrete_value:
                element_info = self.generic_utils.get_instance_generic_info(element)
                if element_info.is_generic and element_info.concrete_args:
                    # Extract the deepest concrete args (e.g., int from Box[int])
                    deepest_args = self._extract_deepest_type_args(element_info)
                    element_types.update(deepest_args)
                else:
                    element_types.add(type(element))
            
            if element_types:
                # Create a union type from all the element types
                union_type = create_union_if_needed(element_types)
                constraints.append(Constraint(typevar, union_type, Variance.COVARIANT))
                return
        
        # For generic objects (dataclasses, Pydantic models, etc.), try to extract from their structure
        value_info = self.generic_utils.get_instance_generic_info(concrete_value)
        if value_info.is_generic and value_info.concrete_args:
            deepest_args = self._extract_deepest_type_args(value_info)
            if deepest_args:
                union_type = create_union_if_needed(set(deepest_args))
                constraints.append(Constraint(typevar, union_type, Variance.COVARIANT))
                return
        
        # Fallback: bind to the type of the concrete value
        concrete_type = type(concrete_value)
        constraints.append(Constraint(typevar, concrete_type, Variance.COVARIANT))
    
    def _extract_deepest_type_args(self, generic_info: Any) -> List[type]:
        """Extract the deepest concrete type arguments from a GenericInfo structure."""
        result = []
        
        if not generic_info.concrete_args:
            # No concrete args - return the origin if it's a concrete type
            if not isinstance(generic_info.origin, TypeVar):
                result.append(generic_info.origin)
            return result
        
        for arg_info in generic_info.concrete_args:
            if isinstance(arg_info.origin, TypeVar):
                # This is still a TypeVar - can't extract concrete type
                continue
            elif arg_info.concrete_args:
                # Recursively extract from deeper structure
                result.extend(self._extract_deepest_type_args(arg_info))
            else:
                # This is a concrete type
                result.append(arg_info.origin)
        
        return result
    
    def _extract_from_nested_structure(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Fallback method to extract constraints from nested structures.
        
        Uses generic_utils to extract type information from both annotations and instances,
        which handles __orig_class__, __pydantic_generic_metadata__, and other type metadata.
        """
        # Extract type information using generic_utils (handles all type metadata internally)
        ann_info = self.generic_utils.get_generic_info(annotation)
        val_info = self.generic_utils.get_instance_generic_info(value)
        
        # If value has generic type information, try to align structures
        if val_info.is_generic and ann_info.concrete_args and val_info.concrete_args:
            if len(ann_info.concrete_args) == len(val_info.concrete_args):
                for ann_arg, val_arg in zip(ann_info.concrete_args, val_info.concrete_args):
                    # Recursively handle nested generic structures
                    self._extract_from_annotation_alignment(ann_arg, val_arg, constraints)
                return
        
        # Try to extract from instance fields for dataclasses and similar
        self._extract_from_instance_fields(annotation, value, constraints)
    
    def _extract_from_annotation_alignment(self, ann_type: Any, val_type: Any, constraints: List[Constraint]):
        """Extract constraints by aligning annotation and value type structures.
        
        Uses generic_utils for all type metadata extraction instead of manual checks.
        """
        ann_info = self.generic_utils.get_generic_info(ann_type)
        val_info = self.generic_utils.get_generic_info(val_type)
        
        # Case 1: Unparameterized generic annotation with parameterized value type
        # (e.g., Box vs Box[int] - extract TypeVars from annotation and match with value's concrete args)
        if ann_info.type_params and not ann_info.concrete_args and val_info.concrete_args:
            if len(ann_info.type_params) == len(val_info.concrete_args):
                for ann_typevar, val_concrete in zip(ann_info.type_params, val_info.concrete_args):
                    if isinstance(ann_typevar, TypeVar):
                        constraints.append(Constraint(ann_typevar, val_concrete))
                return
        
        # Case 2: Both have compatible structure - align their type arguments
        if (self._origins_are_compatible(ann_info.origin, val_info.origin) and 
            ann_info.concrete_args and val_info.concrete_args and 
            len(ann_info.concrete_args) == len(val_info.concrete_args)):
            for ann_arg, val_arg in zip(ann_info.concrete_args, val_info.concrete_args):
                if isinstance(ann_arg, TypeVar):
                    constraints.append(Constraint(ann_arg, val_arg))
                else:
                    # Recursively handle deeper nesting
                    self._extract_from_annotation_alignment(ann_arg, val_arg, constraints)
        
        # Case 3: Direct TypeVar binding
        elif isinstance(ann_type, TypeVar):
            constraints.append(Constraint(ann_type, val_type))
        
        # Case 4: Both types are generic but different structures - try type params alignment
        elif ann_info.type_params and val_info.concrete_args:
            if len(ann_info.type_params) == len(val_info.concrete_args):
                for ann_typevar, val_concrete in zip(ann_info.type_params, val_info.concrete_args):
                    if isinstance(ann_typevar, TypeVar):
                        constraints.append(Constraint(ann_typevar, val_concrete))
    
    def _extract_from_instance_fields(self, annotation: Any, value: Any, constraints: List[Constraint]):
        """Extract type constraints from instance field values."""
        
        # Handle dataclasses
        if is_dataclass(value):
            ann_info = self.generic_utils.get_generic_info(annotation)
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
                        union_type = create_union_if_needed(field_types)
                        constraints.append(Constraint(ann_info.type_params[0], union_type))
        
        # Handle other generic instances by trying to extract from __dict__
        elif hasattr(value, '__dict__'):
            ann_info = self.generic_utils.get_generic_info(annotation)
            if ann_info.type_params and len(ann_info.type_params) == 1:
                # Extract types from instance attributes
                attr_types = {type(v) for v in value.__dict__.values() if v is not None}
                if attr_types:
                    union_type = create_union_if_needed(attr_types)
                    constraints.append(Constraint(ann_info.type_params[0], union_type))
    
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
        context: Substitution
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
            if concrete_type not in typevar.__constraints__:
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
    
    def _try_distribute_union_types(self, values: set, union_alternatives: List, constraints: List[Constraint]) -> bool:
        """
        Try to distribute types from a set among TypeVars in a Union.
        
        For Set[Union[A, B]] with values {1, 'a', 2, 'b'}, distribute types so that
        A=int and B=str (or vice versa), rather than A=int|str and B=int|str.
        
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
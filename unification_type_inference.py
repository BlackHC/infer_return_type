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
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass, field

# Import unified generic utilities
from generic_utils import (
    GenericInfo,
    get_generic_info,
    get_instance_generic_info,
    create_generic_info_union_if_needed,
    get_annotation_value_pairs,
    is_union_type,
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

    def __init__(
        self,
        typevar: TypeVar,
        concrete_type: GenericInfo,
        variance: Variance = Variance.INVARIANT,
        is_override: bool = False,
    ):
        self.typevar = typevar
        self.concrete_type = concrete_type  # Always GenericInfo
        self.variance = variance
        self.is_override = is_override

    @property
    def concrete_type_resolved(self) -> Any:
        """Get the resolved concrete type."""
        return self.concrete_type.resolved_type

    def __str__(self):
        override_str = " (override)" if self.is_override else ""
        return f"{self.typevar} ~ {self.concrete_type_resolved} ({self.variance.value}){override_str}"

    def __repr__(self):
        return self.__str__()


@dataclass
class Substitution:
    """Represents a substitution of TypeVars to concrete types."""

    bindings: Dict[TypeVar, GenericInfo] = field(
        default_factory=dict
    )  # Always GenericInfo

    def bind(self, typevar: TypeVar, concrete_type: GenericInfo):
        """Bind a TypeVar to a concrete type (GenericInfo)."""
        self.bindings[typevar] = concrete_type

    def get(self, typevar: TypeVar) -> Optional[GenericInfo]:
        """Get the binding for a TypeVar."""
        return self.bindings.get(typevar)

    def get_resolved(self, typevar: TypeVar) -> Optional[Any]:
        """Get the resolved binding for a TypeVar."""
        binding = self.bindings.get(typevar)
        if binding is None:
            return None
        return binding.resolved_type

    def apply(self, annotation: Any) -> Any:
        """Apply this substitution to an annotation."""
        annotation_info = get_generic_info(annotation)
        substituted_info = _substitute_typevars_in_generic_info(
            annotation_info, self.bindings
        )
        return substituted_info.resolved_type


def unify_annotation_with_value(
    annotation: Any, value: Any, constraints: List[Constraint] = None
) -> Substitution:
    """
    Unify an annotation with a concrete value to produce TypeVar bindings.

    This is the main entry point for type inference.
    """
    if constraints is None:
        constraints = []

    # Collect constraints from the annotation/value pair
    collect_constraints(annotation, value, constraints)

    # Solve the constraint system
    return solve_constraints(constraints)


def collect_constraints(annotation: type, value: Any, constraints: List[Constraint]):
    """Recursively collect type constraints from annotation/value pairs."""

    try:
        annotation_info = get_generic_info(annotation)
        _collect_constraints_internal(annotation_info, value, constraints)
    except UnificationError as e:
        # Convert to TypeInferenceError for consistency
        raise TypeInferenceError(str(e)) from e


def _collect_constraints_internal(
    annotation_info: GenericInfo, value: Any, constraints: List[Constraint]
):
    """Internal constraint collection that can raise UnificationError."""

    # Base case: Direct TypeVar
    if isinstance(annotation_info.origin, TypeVar):
        concrete_type_info = _infer_type_from_value(value)
        constraints.append(Constraint(annotation_info.origin, concrete_type_info))
        return

    # Special case: Union types (need custom logic for alternative selection)
    if is_union_type(annotation_info.origin):
        _handle_union_annotation(annotation_info, value, constraints)
        return

    # Try unified extraction first for standard containers and custom types
    # Use GenericInfo directly instead of converting to resolved_type
    pairs = get_annotation_value_pairs(annotation_info.resolved_type, value)
    if pairs:
        _collect_constraints_from_pairs(pairs, constraints)
        return

    # Fallback: Try direct generic structure matching for custom types
    # This handles cases like Pydantic[A, list[B]] with Pydantic[int, list[str]]
    val_info = get_instance_generic_info(value)
    if _match_generic_structures(annotation_info, val_info, constraints):
        return

    # Final fallback: handle non-container types or types without type parameters
    if annotation_info.origin and annotation_info.origin is not type(value):
        # Special case: ForwardRef that couldn't be resolved
        # Try to match by name with value's type
        if hasattr(annotation_info.origin, "__forward_arg__"):
            # It's a ForwardRef - check if value's class name matches
            value_class_name = type(value).__name__
            forward_name = annotation_info.origin.__forward_arg__
            # Extract class name from ForwardRef (e.g., 'TreeNode[A]' → 'TreeNode')
            ref_class_name = (
                forward_name.split("[")[0] if "[" in forward_name else forward_name
            )
            if value_class_name == ref_class_name:
                # Names match - use direct structure matching with instance
                val_info = get_instance_generic_info(value)
                if _match_generic_structures(annotation_info, val_info, constraints):
                    return
                # If structure matching didn't work, just skip (assume match by name)
                return

        # Provide more specific error messages for common container mismatches
        origin_name = (
            annotation_info.origin.__name__
            if hasattr(annotation_info.origin, "__name__")
            else str(annotation_info.origin)
        )
        value_type_name = type(value).__name__
        raise UnificationError(f"Expected {origin_name}, got {value_type_name}")


def _collect_constraints_from_pairs(
    pairs: List[Tuple[GenericInfo, Any]], constraints: List[Constraint]
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
            _add_covariant_constraints_for_elements(typevar, values, constraints)

    # Handle complex cases (non-TypeVar pairs)
    if complex_pairs:
        _process_complex_pairs(complex_pairs, constraints)


def _process_complex_pairs(
    complex_pairs: List[Tuple[GenericInfo, Any]], constraints: List[Constraint]
):
    """Process complex (non-TypeVar) annotation-value pairs."""

    # Check if we have multiple pairs with the same Union type for distribution
    first_info, _ = complex_pairs[0]

    # Check if the first pair is itself a Union type
    if is_union_type(first_info.origin):
        # Check if all pairs have the same Union structure
        all_same_union = all(is_union_type(gi.origin) for gi, _ in complex_pairs)

        if all_same_union:
            # All elements are Union types - handle them
            union_args = first_info.concrete_args

            # For multiple values, try distribution first
            if len(complex_pairs) > 1:
                all_values = [val for _, val in complex_pairs]
                if _try_distribute_union_types(all_values, union_args, constraints):
                    return

            # Match each value to union alternatives
            for _, val in complex_pairs:
                _match_value_to_union_alternatives(val, union_args, constraints)
            return

    # For non-union pairs, process each individually
    for generic_info, val in complex_pairs:
        # Use GenericInfo-based matching instead of resolved_type
        # to avoid losing TypeVars due to Pydantic same-TypeVar optimization
        val_info = get_instance_generic_info(val)
        if _match_generic_structures(generic_info, val_info, constraints):
            continue
        # Fallback: recursively collect constraints
        _collect_constraints_internal(generic_info, val, constraints)


def _add_covariant_constraints_for_elements(
    typevar: TypeVar, values, constraints: List[Constraint]
):
    """
    Add separate covariant constraints for each distinct type in values.

    This allows proper union formation and bound checking in the constraint solver.
    Uses _infer_type_from_value to get full generic types, not just base types.
    """
    element_types = {_infer_type_from_value(item) for item in values}
    for element_type in element_types:
        constraints.append(Constraint(typevar, element_type, Variance.COVARIANT))


def _match_generic_structures(
    annotation_info: GenericInfo, value_info: GenericInfo, constraints: List[Constraint]
) -> bool:
    """Match two GenericInfo objects to extract TypeVar bindings.

    This avoids the Pydantic same-TypeVar optimization issue where Box[A].resolved_type
    returns Box instead of Box[A], losing the TypeVar information.

    Args:
        annotation_info: GenericInfo from the annotation
        value_info: GenericInfo from the instance
        constraints: List to append constraints to

    Returns:
        True if any constraints were found, False otherwise
    """
    # Both must be generic with compatible origins
    if not (annotation_info.is_generic and value_info.is_generic):
        return False

    if not _origins_compatible(annotation_info.origin, value_info.origin):
        return False

    if len(annotation_info.concrete_args) != len(value_info.concrete_args):
        return False

    # Match each type argument pair
    found_constraints = False
    for ann_arg, val_arg in zip(
        annotation_info.concrete_args, value_info.concrete_args
    ):
        # Direct TypeVar binding
        if isinstance(ann_arg.origin, TypeVar) and not isinstance(
            val_arg.origin, TypeVar
        ):
            constraints.append(Constraint(ann_arg.origin, val_arg, Variance.INVARIANT))
            found_constraints = True
        # Recursive matching for nested structures
        elif ann_arg.is_generic and val_arg.is_generic:
            if _match_generic_structures(ann_arg, val_arg, constraints):
                found_constraints = True

    return found_constraints


def _origins_compatible(origin1: type, origin2: type) -> bool:
    """Check if two origins are compatible for type matching.

    This is more flexible than simple equality, allowing for:
    - Exact matches: list == list
    - Generic vs base: list[int] and list (same origin)
    """
    if origin1 is origin2:
        return True

    # Both are the same base type
    if origin1 == origin2:
        return True

    return False


def _handle_union_annotation(
    annotation_info: GenericInfo, value: Any, constraints: List[Constraint]
):
    """Handle Union type annotations by trying each alternative and picking the best match."""
    args = annotation_info.concrete_args

    # Try each union alternative
    best_constraints = None
    best_score = -1

    for alternative_info in args:
        try:
            temp_constraints = []
            if isinstance(alternative_info.origin, TypeVar):
                # Direct TypeVar alternative - use inferred type from value
                concrete_info = _infer_type_from_value(value)
                temp_constraints.append(
                    Constraint(
                        alternative_info.origin, concrete_info, Variance.INVARIANT
                    )
                )
            else:
                # Use GenericInfo directly - no need to convert back and forth
                _collect_constraints_internal(alternative_info, value, temp_constraints)

            # Score this alternative - prefer structured matches over direct TypeVar matches
            # Structured matches (like List[A], Dict[K,V]) provide more specific constraints
            score = len(temp_constraints)

            # Bonus points for matching structured types (not just bare TypeVar)
            if not isinstance(alternative_info.origin, TypeVar):
                # Check if the alternative structure matches the value structure
                alt_origin = alternative_info.origin
                value_type = type(value)
                if _origins_compatible(alt_origin, value_type):
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
        raise UnificationError(
            f"Value {value} doesn't match any alternative in {annotation_info.resolved_type}"
        )


def _try_distribute_union_types(
    values, union_alternatives: List[GenericInfo], constraints: List[Constraint]
) -> bool:
    """
    Try to distribute types from values among TypeVars in a Union.

    For Set[Union[A, B]] with values {1, 'a', 2, 'b'}, distribute types so that
    A=int and B=str (or vice versa), rather than A=int|str and B=int|str.

    Returns True if distribution was successful, False otherwise.
    """
    # Only works if all union alternatives are TypeVars
    typevars = [
        alt.origin for alt in union_alternatives if isinstance(alt.origin, TypeVar)
    ]
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
            concrete_info = get_generic_info(concrete_type)
            constraints.append(Constraint(typevar, concrete_info, Variance.INVARIANT))

        return True

    # Can't distribute evenly, fall back to default behavior
    return False


def _match_value_to_union_alternatives(
    value: Any, union_alternatives: List[GenericInfo], constraints: List[Constraint]
):
    """Match a value against union alternatives and collect constraints.

    This implements context-aware TypeVar matching for Union types.
    """
    value_type = type(value)

    # First, check if the value exactly matches any concrete (non-TypeVar) type in the union
    # This handles cases like Optional[A] where None should match the concrete None type
    for alt_info in union_alternatives:
        if (
            not isinstance(alt_info.origin, TypeVar)
            and alt_info.resolved_type == value_type
        ):
            # Perfect match with concrete type - no constraints needed
            return

    # Try to match value against complex generic alternatives (e.g., Dict[str, A] in Optional[Dict[str, A]])
    # This is crucial for handling Optional[ComplexType[A]] patterns
    for alt_info in union_alternatives:
        if not isinstance(alt_info.origin, TypeVar):
            # This is a structured type (like Dict[str, A]) - try to match it
            try:
                temp_constraints = []
                # alt_info is already GenericInfo, no need to convert
                _collect_constraints_internal(alt_info, value, temp_constraints)
                # If we successfully collected constraints, use them
                if temp_constraints:
                    constraints.extend(temp_constraints)
                    return
            except (UnificationError, TypeError):
                # This alternative doesn't match - try next
                continue

    # Get existing TypeVar bindings for context-aware matching
    existing_bindings = _get_existing_typevar_bindings(constraints)

    # Try to assign this value to the TypeVar that already has evidence for this type
    matched_typevar = None
    for alt_info in union_alternatives:
        if (
            isinstance(alt_info.origin, TypeVar)
            and alt_info.origin in existing_bindings
        ):
            existing_types = existing_bindings[alt_info.origin]
            if len(existing_types) == 1 and value_type in existing_types:
                # Perfect match - this TypeVar already has evidence for this exact type
                matched_typevar = alt_info.origin
                break

    if matched_typevar:
        # Add a covariant constraint since this is coming from a Set/collection
        value_info = get_generic_info(value_type)
        constraints.append(Constraint(matched_typevar, value_info, Variance.COVARIANT))
        return

    # No perfect match - check if we can rule out some TypeVars based on conflicting evidence
    ruled_out = set()
    for alt_info in union_alternatives:
        if (
            isinstance(alt_info.origin, TypeVar)
            and alt_info.origin in existing_bindings
        ):
            existing_types = existing_bindings[alt_info.origin]
            if len(existing_types) == 1 and value_type not in existing_types:
                # This TypeVar has strong evidence for a different type
                ruled_out.add(alt_info.origin)

    # Add constraints for remaining candidates
    candidates = [
        alt_info.origin
        for alt_info in union_alternatives
        if isinstance(alt_info.origin, TypeVar) and alt_info.origin not in ruled_out
    ]

    if candidates:
        # Use covariant constraints to allow union formation if needed
        for candidate in candidates:
            value_info = get_generic_info(value_type)
            constraints.append(Constraint(candidate, value_info, Variance.COVARIANT))
    else:
        # Fallback: add constraints for all TypeVar alternatives with invariant variance
        for alt_info in union_alternatives:
            if isinstance(alt_info.origin, TypeVar):
                value_info = get_generic_info(value_type)
                constraints.append(
                    Constraint(alt_info.origin, value_info, Variance.INVARIANT)
                )


def _get_existing_typevar_bindings(
    constraints: List[Constraint], variance_filter: Variance = Variance.INVARIANT
) -> Dict[TypeVar, Set[GenericInfo]]:
    """Extract existing TypeVar bindings from constraints for context-aware resolution.

    Args:
        constraints: List of constraints to analyze
        variance_filter: Only consider constraints with this variance (default: INVARIANT for strong evidence)

    Returns:
        Dictionary mapping TypeVars to sets of GenericInfo they're constrained to
    """
    bindings = defaultdict(set)
    for constraint in constraints:
        if constraint.variance == variance_filter:
            bindings[constraint.typevar].add(constraint.concrete_type)
    return dict(bindings)


def solve_constraints(constraints: List[Constraint]) -> Substitution:
    """Solve the constraint system to produce a substitution with global context awareness."""

    substitution = Substitution()

    # Group constraints by TypeVar
    constraint_groups = defaultdict(list)
    for constraint in constraints:
        constraint_groups[constraint.typevar].append(constraint)

    # First pass: resolve TypeVars with unambiguous constraints
    resolved_in_first_pass = set()

    for typevar, typevar_constraints in constraint_groups.items():
        if _can_resolve_unambiguously(typevar_constraints):
            resolved_type = _resolve_typevar_constraints(typevar, typevar_constraints)
            substitution.bind(typevar, resolved_type)
            resolved_in_first_pass.add(typevar)

    # Second pass: resolve remaining TypeVars
    for typevar, typevar_constraints in constraint_groups.items():
        if typevar not in resolved_in_first_pass:
            resolved_type = _resolve_typevar_constraints(typevar, typevar_constraints)
            substitution.bind(typevar, resolved_type)

    return substitution


def _can_resolve_unambiguously(constraints: List[Constraint]) -> bool:
    """Check if constraints can be resolved without ambiguity."""
    if len(constraints) <= 1:
        return True

    # If all constraints have the same concrete type, unambiguous
    concrete_types = [c.concrete_type_resolved for c in constraints]
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


def _resolve_typevar_constraints(
    typevar: TypeVar, constraints: List[Constraint]
) -> GenericInfo:
    """Resolve constraints for a single TypeVar."""

    if len(constraints) == 1:
        constraint = constraints[0]
        resolved_type = _check_typevar_bounds(typevar, constraint.concrete_type)
        # resolved_type is already a GenericInfo from _check_typevar_bounds
        return resolved_type

    # Check if we have any override constraints
    override_constraints = [c for c in constraints if c.is_override]

    # If we have override constraints, they take precedence
    if override_constraints:
        if len(override_constraints) == 1:
            # Single override - use it
            resolved_type = _check_typevar_bounds(
                typevar, override_constraints[0].concrete_type
            )
            return resolved_type
        else:
            # Multiple overrides - they must be consistent
            override_types = [c.concrete_type_resolved for c in override_constraints]
            if len(set(override_types)) == 1:
                resolved_type = _check_typevar_bounds(
                    typevar, override_constraints[0].concrete_type
                )
                return resolved_type
            else:
                raise UnificationError(
                    f"Conflicting override constraints for {typevar}: {override_constraints}"
                )

    # No overrides - handle normally
    concrete_types = [c.concrete_type_resolved for c in constraints]

    # Check if all constraints are the same
    if len(set(concrete_types)) == 1:
        resolved_type = _check_typevar_bounds(typevar, constraints[0].concrete_type)
        return resolved_type

    # Different constraints - form union
    covariant_constraints = [c for c in constraints if c.variance == Variance.COVARIANT]
    invariant_constraints = [c for c in constraints if c.variance == Variance.INVARIANT]

    # If we have covariant constraints (like List[A] with mixed elements), form union
    if covariant_constraints and not invariant_constraints:
        # Use GenericInfo objects directly for union formation
        concrete_type_infos = [c.concrete_type for c in covariant_constraints]
        union_info = create_generic_info_union_if_needed(set(concrete_type_infos))
        resolved_type = _check_typevar_bounds(typevar, union_info)
        return resolved_type

    # If we have multiple invariant constraints with different types, form a union
    # This handles cases like: def identity(a: A, b: A) -> A with identity(1, 'x')
    if len(invariant_constraints) > 1:
        invariant_types = [c.concrete_type_resolved for c in invariant_constraints]
        if len(set(invariant_types)) > 1:
            # Multiple independent sources with different types - create union
            # Use GenericInfo objects directly for union formation
            concrete_type_infos = [c.concrete_type for c in invariant_constraints]
            union_info = create_generic_info_union_if_needed(set(concrete_type_infos))
            resolved_type = _check_typevar_bounds(typevar, union_info)
            return resolved_type

    # Mixed variance - default to union formation
    # Use GenericInfo objects directly for union formation
    concrete_type_infos = [c.concrete_type for c in constraints]
    union_info = create_generic_info_union_if_needed(set(concrete_type_infos))
    resolved_type = _check_typevar_bounds(typevar, union_info)
    return resolved_type


def _check_typevar_bounds(
    typevar: TypeVar, concrete_type_info: GenericInfo
) -> GenericInfo:
    """Check if concrete type satisfies TypeVar bounds and constraints.

    Per PEP 484, constrained TypeVars must resolve to exactly ONE of the specified types,
    not a union of them. Union types are rejected for constrained TypeVars.
    """

    # concrete_type_info is already a GenericInfo
    type_info = concrete_type_info
    origin = type_info.origin

    # Check explicit constraints (e.g., TypeVar('T', int, str))
    if typevar.__constraints__:
        # For union types, check if it matches any constraint (which may also be unions)
        if is_union_type(origin):
            # Check if the union matches any of the constraint unions
            if not _matches_any_constraint(concrete_type_info, typevar.__constraints__):
                raise UnificationError(
                    f"Type {concrete_type_info.resolved_type} violates constraints {typevar.__constraints__} for {typevar}"
                )
        # For non-union types, check direct match or origin match
        elif not _matches_any_constraint(concrete_type_info, typevar.__constraints__):
            raise UnificationError(
                f"Type {concrete_type_info.resolved_type} violates constraints {typevar.__constraints__} for {typevar}"
            )

    # Check bound (e.g., TypeVar('T', bound=int))
    if typevar.__bound__:
        # For union types, check if all components satisfy the bound
        if is_union_type(origin):
            union_args = type_info.concrete_args
            # All components must satisfy the bound
            for arg_info in union_args:
                if not _is_subtype(arg_info.resolved_type, typevar.__bound__):
                    raise UnificationError(
                        f"Type {arg_info.resolved_type} in union {concrete_type_info.resolved_type} doesn't satisfy bound {typevar.__bound__} for {typevar}"
                    )
        else:
            # Single type must satisfy the bound
            if not _is_subtype(concrete_type_info.resolved_type, typevar.__bound__):
                raise UnificationError(
                    f"Type {concrete_type_info.resolved_type} doesn't satisfy bound {typevar.__bound__} for {typevar}"
                )

    # Return the GenericInfo
    return concrete_type_info


def _matches_any_constraint(
    inferred_type_info: GenericInfo, constraints: tuple
) -> bool:
    """Check if inferred type matches any of the constraints.

    Handles:
    - Exact match: int == int
    - Origin match: list[int] matches list
    - Union match: (list[int] | float) matches (float | list) by comparing origins
    """
    # inferred_type_info is already a GenericInfo
    inferred_info = inferred_type_info

    # Check each constraint
    for constraint in constraints:
        # Exact match
        if inferred_info.resolved_type == constraint:
            return True

        # Origin match for generic types (list[int] matches list)
        constraint_info = get_generic_info(constraint)
        if _origins_compatible(inferred_info.origin, constraint_info.origin):
            # If constraint is bare type (list) and inferred is specialized (list[int]), accept it
            if not constraint_info.is_generic and inferred_info.is_generic:
                return True
            # If both have same structure, check recursively (handled by exact match above)
            elif inferred_info.is_generic and constraint_info.is_generic:
                # Both generic - check if they're equivalent
                if inferred_info.resolved_type == constraint:
                    return True

        # Union constraint matching: check if inferred union components match constraint union components
        if is_union_type(inferred_info.origin) and is_union_type(
            constraint_info.origin
        ):
            # Both are unions - check if components match by origin
            if _union_components_match(inferred_info, constraint_info):
                return True

    return False


def _union_components_match(
    inferred_union_info: GenericInfo, constraint_union_info: GenericInfo
) -> bool:
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


def _is_subtype(subtype: type, supertype: type) -> bool:
    """Check if subtype is a subtype of supertype."""
    try:
        return issubclass(subtype, supertype)
    except TypeError:
        # Handle cases where subtype might not be a class
        return False


def _infer_type_from_value(value: Any) -> GenericInfo:
    """Infer the most specific type from a value as GenericInfo."""
    if value is None:
        return GenericInfo(origin=type(None))

    base_type = type(value)

    # For collections, try to infer element types
    if isinstance(value, list) and value:
        element_types = {type(item) for item in value}
        # Create GenericInfo objects for each element type and form union
        element_infos = [GenericInfo(origin=t) for t in element_types]
        union_info = create_generic_info_union_if_needed(set(element_infos))
        return GenericInfo(origin=list, concrete_args=[union_info])
    elif isinstance(value, dict) and value:
        key_types = {type(k) for k in value.keys()}
        value_types = {type(v) for v in value.values()}

        # Handle key types
        key_infos = [GenericInfo(origin=t) for t in key_types]
        key_info = create_generic_info_union_if_needed(set(key_infos))

        # Handle value types
        value_infos = [GenericInfo(origin=t) for t in value_types]
        value_info = create_generic_info_union_if_needed(set(value_infos))
        return GenericInfo(origin=dict, concrete_args=[key_info, value_info])
    elif isinstance(value, tuple):
        element_types = tuple(type(item) for item in value)
        element_infos = [GenericInfo(origin=t) for t in element_types]
        return GenericInfo(origin=tuple, concrete_args=element_infos)
    elif isinstance(value, set) and value:
        element_types = {type(item) for item in value}
        # Create GenericInfo objects for each element type and form union
        element_infos = [GenericInfo(origin=t) for t in element_types]
        union_info = create_generic_info_union_if_needed(set(element_infos))
        return GenericInfo(origin=set, concrete_args=[union_info])

    return GenericInfo(origin=base_type)


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

    # Collect all constraints from function parameters
    all_constraints = []

    # Process positional arguments
    param_names = list(sig.parameters.keys())
    for i, arg in enumerate(args):
        if i < len(param_names):
            param = sig.parameters[param_names[i]]
            if param.annotation != inspect.Parameter.empty:
                collect_constraints(param.annotation, arg, all_constraints)

    # Process keyword arguments
    for name, value in kwargs.items():
        if name in sig.parameters:
            param = sig.parameters[name]
            if param.annotation != inspect.Parameter.empty:
                collect_constraints(param.annotation, value, all_constraints)

    # Add type overrides as constraints
    for typevar, override_type in type_overrides.items():
        override_info = get_generic_info(override_type)
        all_constraints.append(Constraint(typevar, override_info, is_override=True))

    # Solve constraints to get substitution
    try:
        substitution = solve_constraints(all_constraints)
    except UnificationError as e:
        raise TypeInferenceError(str(e)) from e

    # Apply substitution to return annotation using GenericInfo pipeline
    return_info = get_generic_info(return_annotation)
    substituted_info = _substitute_typevars_in_generic_info(
        return_info, substitution.bindings
    )

    # Convert to final type using resolved_type
    result = substituted_info.resolved_type

    # Check for any remaining unbound TypeVars in complex types
    if _has_unbound_typevars_in_generic_info(substituted_info):
        raise TypeInferenceError(
            f"Could not fully infer return type - some TypeVars remain unbound: {result}"
        )

    return result


def _substitute_typevars_in_generic_info(
    generic_info: GenericInfo, bindings: Dict[TypeVar, GenericInfo]
) -> GenericInfo:
    """Substitute TypeVars in a GenericInfo structure with their bindings."""

    # If this GenericInfo represents a TypeVar, try to substitute it
    if isinstance(generic_info.origin, TypeVar):
        if generic_info.origin in bindings:
            bound_value = bindings[generic_info.origin]
            # bound_value is always GenericInfo now
            return bound_value
        else:
            # Unbound TypeVar - return as-is
            return generic_info

    # Handle Union types specially - only include bound TypeVars
    if is_union_type(generic_info.origin):
        substituted_args = []

        for arg_info in generic_info.concrete_args:
            substituted_arg = _substitute_typevars_in_generic_info(arg_info, bindings)
            # Only include the arg if it doesn't contain unbound TypeVars
            if not _has_unbound_typevars_in_generic_info(substituted_arg):
                substituted_args.append(substituted_arg)

        # If we have at least one bound arg, return the union of bound args
        if substituted_args:
            if len(substituted_args) == 1:
                return substituted_args[0]
            # Create union GenericInfo
            return GenericInfo(
                origin=generic_info.origin, concrete_args=substituted_args
            )

        # If no args were bound, return the original annotation (will be caught as unbound)
        return generic_info

    # Recursively substitute in concrete_args
    substituted_args = []
    for arg_info in generic_info.concrete_args:
        substituted_args.append(
            _substitute_typevars_in_generic_info(arg_info, bindings)
        )

    # Create new GenericInfo with substituted arguments
    return GenericInfo(origin=generic_info.origin, concrete_args=substituted_args)


def _has_unbound_typevars_in_generic_info(generic_info: GenericInfo) -> bool:
    """Check if a GenericInfo contains any unbound TypeVars."""
    if isinstance(generic_info.origin, TypeVar):
        return True

    # Recursively check concrete_args
    for arg_info in generic_info.concrete_args:
        if _has_unbound_typevars_in_generic_info(arg_info):
            return True

    return False

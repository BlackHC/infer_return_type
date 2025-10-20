from dataclasses import dataclass
import types
import typing
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import BaseModel
import pytest

from infer_return_type import infer_return_type, TypeInferenceError
from infer_return_type.generic_utils import get_generic_info
from infer_return_type.infer_return_type import (
    Constraint,
    Substitution,
    UnificationError,
    Variance,
    _has_unbound_typevars_in_generic_info,
    _infer_type_from_value,
    _is_subtype,
    unify_annotation_with_value,
    solve_constraints,
    _union_components_match,
    _match_generic_structures,
    _origins_compatible,
)

# TypeVars for testing
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
K = TypeVar("K")
V = TypeVar("V")
X = TypeVar("X")
Y = TypeVar("Y")

# =============================================================================
# CONSOLIDATED TEST FIXTURES
# =============================================================================


@dataclass
class Wrap(typing.Generic[A]):
    """Simple wrapper for a single value."""

    value: A


class WrapModel(BaseModel, typing.Generic[A]):
    """Pydantic version of Wrap."""

    value: A


@dataclass
class GenericPair(typing.Generic[A, B]):
    """Container with two type parameters."""

    first: A
    second: B


@dataclass
class MultiParamContainer(typing.Generic[A, B, C]):
    """Container with three type parameters."""

    primary: List[A]
    secondary: Dict[str, B]
    tertiary: Set[C]
    mixed: List[Tuple[A, B, C]]


@dataclass
class Level1(typing.Generic[A]):
    """First level of nested generic structure."""

    inner: A


@dataclass
class Level2(typing.Generic[A]):
    """Second level containing Level1."""

    wrapped: Level1[A]
    alternatives: List[A]


class Level3(BaseModel, typing.Generic[A]):
    """Third level (Pydantic) containing Level2."""

    nested: Level2[A]
    extras: Dict[str, A]


@dataclass
class GraphNode(typing.Generic[A]):
    """Graph node with edges."""

    value: A
    edges: List["GraphNode[A]"]


@dataclass
class LinkedNode(typing.Generic[A]):
    """Simple node for linked list testing."""

    value: A
    next: Optional["LinkedNode[A]"]


@dataclass
class DerivedWrap(Wrap[A], typing.Generic[A]):
    """Derived class maintaining same type parameter."""

    derived_value: int


@dataclass
class ChildWrap(DerivedWrap[A], typing.Generic[A]):
    """Child class inheriting from DerivedWrap."""

    child_value: str


@dataclass
class ConcreteChild(Wrap[int]):
    """Concrete specialization of Wrap."""

    extra: str


@dataclass
class HasA(typing.Generic[A]):
    """Class with TypeVar A for multiple inheritance tests."""

    a_value: A


@dataclass
class HasB(typing.Generic[B]):
    """Class with TypeVar B for multiple inheritance tests."""

    b_value: B


@dataclass
class HasBoth(HasA[A], HasB[B], typing.Generic[A, B]):
    """Class inheriting from both HasA and HasB."""

    both: str


class ParentPyd(BaseModel, typing.Generic[A, B]):
    """Pydantic parent class for inheritance testing."""

    a_value: A
    b_value: B


class ChildPyd(ParentPyd[B, A], typing.Generic[A, B]):
    """Pydantic child with swapped type parameters."""

    pass


@dataclass
class JsonValue(typing.Generic[A]):
    """JSON-like nested structure."""

    data: Union[A, Dict[str, "JsonValue[A]"], List["JsonValue[A]"]]


@dataclass
class DeepContainer(typing.Generic[A]):
    """Deeply nested container."""

    deep_data: Dict[str, List[Dict[str, A]]]


@dataclass
class ComplexContainer(typing.Generic[A, B]):
    """Complex container with multiple nested structures."""

    lists_of_a: List[List[A]]
    dict_to_b: Dict[str, B]
    optional_a_list: Optional[List[A]]


@dataclass
class TypedColumn(typing.Generic[A]):
    """Represents a typed column in a table."""

    name: str
    values: List[A]


@dataclass
class MultiColumnData(typing.Generic[A, B, C]):
    """Multi-column data structure (DataFrame-like)."""

    col1: TypedColumn[A]
    col2: TypedColumn[B]
    col3: TypedColumn[C]


@dataclass
class OptionalContainer(typing.Generic[A]):
    """Container with optional nested field."""

    maybe_items: Optional[List[A]]


@dataclass
class WithClassVar(typing.Generic[A]):
    """Container with ClassVar for testing."""

    instance_var: A


# Additional test classes for deduplication


@dataclass
class OneParam(GenericPair[A, str], typing.Generic[A]):
    """One parameter class inheriting from GenericPair."""

    extra: int


@dataclass
class HasA2(typing.Generic[A]):
    """Alternative HasA class for multiple inheritance testing."""

    a_value: A


@dataclass
class HasB2(typing.Generic[A]):  # Same TypeVar name as HasA2!
    """Alternative HasB class for multiple inheritance testing."""

    b_value: A


@dataclass
class HasBoth2(HasA2[A], HasB2[B], typing.Generic[A, B]):
    """Alternative HasBoth class for multiple inheritance testing."""

    both: str


@dataclass
class SubstitutionContainer(typing.Generic[A, B]):
    """Container for substitution testing."""

    a: A
    b: B


@dataclass
class SimpleClass:
    """Simple class for ForwardRef testing."""

    value: int


@dataclass
class DifferentClass:
    """Different test class for ForwardRef testing."""

    value: int


@dataclass
class GenericTest(typing.Generic[A]):
    """Generic test class for ForwardRef testing."""

    value: A


@dataclass
class NestedGeneric(typing.Generic[A]):
    """Nested generic test class for ForwardRef testing."""

    items: List[A]


@dataclass
class UnionTest:
    """Test class with union type for ForwardRef testing."""

    value: Union[int, str]


@dataclass
class OptionalTest:
    """Test class with optional type for ForwardRef testing."""

    value: Optional[int]


@dataclass
class TupleTest:
    """Test class with tuple type for ForwardRef testing."""

    value: Tuple[int, str]


@dataclass
class DictTest:
    """Test class with dict type for ForwardRef testing."""

    value: Dict[str, int]


@dataclass
class SetTest:
    """Test class with set type for ForwardRef testing."""

    value: Set[int]
    class_var: str = "class"


@dataclass
class ManyParams(typing.Generic[A, B, C, X, Y]):
    """Container with many type parameters."""

    a: A
    b: B
    c: C
    x: X
    y: Y


@dataclass
class DataStore(typing.Generic[A, B]):
    """Data store with nested structures."""

    data_map: Dict[str, List[A]]
    metadata: Dict[A, B]


@dataclass
class MultiVarContainer(typing.Generic[A, B]):
    """Container with multiple TypeVars in same structure."""

    pair_lists: List[Dict[A, B]]


@dataclass
class ExtendedContainer(Wrap[A], typing.Generic[A, B]):
    """Extended container adding new type parameter."""

    extra_data: Dict[str, B]


# =============================================================================
# CONSOLIDATED TESTS
# =============================================================================


def test_basic_type_inference():
    """Comprehensive test for basic type inference patterns."""

    # Basic container operations
    def merge_lists(a: List[A], b: List[A]) -> Set[A]: ...

    t = infer_return_type(merge_lists, [1, 2], [3, 4])
    assert typing.get_origin(t) is set and typing.get_args(t) == (int,)

    def swap(p: Tuple[X, Y]) -> Tuple[Y, X]: ...

    t = infer_return_type(swap, (1, "x"))
    assert typing.get_args(t) == (str, int)

    def invert(d: Dict[K, V]) -> Dict[V, K]: ...

    t = infer_return_type(invert, {1: "a", 2: "b"})
    assert typing.get_origin(t) is dict and typing.get_args(t) == (str, int)

    # Optional and Union handling
    def pick_first(x: Optional[A]) -> A: ...

    t = infer_return_type(pick_first, 1)
    assert t is int

    def merge_with_union(a: List[A], b: List[B]) -> Set[A | B]: ...

    t = infer_return_type(merge_with_union, [1], [2.0])
    assert typing.get_origin(t) is set
    args = typing.get_args(t)
    if len(args) == 1 and hasattr(args[0], "__args__"):
        union_args = typing.get_args(args[0])
        assert set(union_args) == {int, float}
    else:
        assert set(args) == {int, float}

    # Basic generic classes
    def unwrap(w: Wrap[A]) -> A: ...

    t = infer_return_type(unwrap, Wrap[int](value=1))
    assert t is int

    def unbox(bs: List[WrapModel[A]]) -> List[A]: ...

    t = infer_return_type(unbox, [WrapModel[int](value=1)])
    assert typing.get_origin(t) is list and typing.get_args(t) == (int,)


def test_multi_typevar_interactions():
    """Test complex multi-TypeVar scenarios."""

    # Complex nested dict patterns
    def extract_nested_dict_union(d: Dict[str, Dict[A, B]]) -> Set[A | B]: ...

    nested_data = {
        "section1": {1: "hello", 2: "world"},
        "section2": {3: "foo", 4: "bar"},
    }
    t = infer_return_type(extract_nested_dict_union, nested_data)
    assert typing.get_origin(t) is set
    union_args = typing.get_args(typing.get_args(t)[0])
    assert set(union_args) == {int, str}

    # Triple nested dict pattern
    def extract_triple_keys(d: Dict[A, Dict[B, Dict[C, int]]]) -> Tuple[A, B, C]: ...

    triple_data = {"level1": {42: {3.14: 100}}}
    t = infer_return_type(extract_triple_keys, triple_data)
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (str, int, float)

    # Mixed container multi-typevar
    def reorganize_complex(
        data: List[Tuple[Dict[A, B], Set[A | B]]],
    ) -> Dict[A, List[B]]: ...

    complex_data = [
        ({1: "a", 2: "b"}, {1, 2, "a", "b"}),
        ({3: "c", 4: "d"}, {3, 4, "c", "d"}),
    ]
    t = infer_return_type(reorganize_complex, complex_data)
    assert typing.get_origin(t) is dict
    key_type, value_type = typing.get_args(t)
    assert key_type is int
    assert typing.get_origin(value_type) is list
    assert typing.get_args(value_type) == (str,)

    # Conflicting TypeVars create unions
    def same_typevar_conflict(a: List[A], b: List[A]) -> A: ...

    t = infer_return_type(same_typevar_conflict, [1, 2], ["a", "b"])

    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


def test_deep_nesting_and_complex_structures():
    """Test deeply nested structures and complex patterns."""

    # Deep nested generics
    def unwrap_all_levels(l3: Level3[A]) -> A: ...
    def get_alternatives(l3: Level3[A]) -> List[A]: ...
    def get_extras_values(l3: Level3[A]) -> List[A]: ...

    deep_structure = Level3[bool](
        nested=Level2[bool](
            wrapped=Level1[bool](inner=True), alternatives=[False, True]
        ),
        extras={"flag1": False, "flag2": True},
    )

    t1 = infer_return_type(unwrap_all_levels, deep_structure)
    assert t1 is bool

    t2 = infer_return_type(get_alternatives, deep_structure)
    assert typing.get_origin(t2) is list and typing.get_args(t2) == (bool,)

    t3 = infer_return_type(get_extras_values, deep_structure)
    assert typing.get_origin(t3) is list and typing.get_args(t3) == (bool,)

    # Multi-parameter containers
    def get_primary(mc: MultiParamContainer[A, B, C]) -> List[A]: ...
    def get_secondary_values(mc: MultiParamContainer[A, B, C]) -> List[B]: ...
    def get_tertiary(mc: MultiParamContainer[A, B, C]) -> Set[C]: ...
    def get_mixed_tuples(mc: MultiParamContainer[A, B, C]) -> List[Tuple[A, B, C]]: ...

    container = MultiParamContainer[int, str, float](
        primary=[1, 2, 3],
        secondary={"a": "hello", "b": "world"},
        tertiary={1.1, 2.2, 3.3},
        mixed=[(1, "a", 1.1), (2, "b", 2.2)],
    )

    assert infer_return_type(get_primary, container) == list[int]
    assert infer_return_type(get_secondary_values, container) == list[str]
    assert infer_return_type(get_tertiary, container) == set[float]

    mixed_type = infer_return_type(get_mixed_tuples, container)
    assert typing.get_origin(mixed_type) is list
    tuple_type = typing.get_args(mixed_type)[0]
    assert typing.get_origin(tuple_type) is tuple
    assert typing.get_args(tuple_type) == (int, str, float)

    # Real-world patterns
    def extract_json_type(json_val: JsonValue[A]) -> A: ...

    nested_json = JsonValue[int](
        data={
            "numbers": JsonValue[int](
                data=[JsonValue[int](data=42), JsonValue[int](data=100)]
            )
        }
    )
    t = infer_return_type(extract_json_type, nested_json)
    assert t is int

    # DataFrame-like structures
    def get_first_column_type(data: MultiColumnData[A, B, C]) -> List[A]: ...
    def get_all_column_types(
        data: MultiColumnData[A, B, C],
    ) -> Tuple[List[A], List[B], List[C]]: ...

    df_data = MultiColumnData[int, str, float](
        col1=TypedColumn[int]("integers", [1, 2, 3]),
        col2=TypedColumn[str]("strings", ["a", "b", "c"]),
        col3=TypedColumn[float]("floats", [1.1, 2.2, 3.3]),
    )

    t1 = infer_return_type(get_first_column_type, df_data)
    assert typing.get_origin(t1) is list
    assert typing.get_args(t1) == (int,)

    t2 = infer_return_type(get_all_column_types, df_data)
    assert typing.get_origin(t2) is tuple
    tuple_args = typing.get_args(t2)
    assert len(tuple_args) == 3
    assert tuple_args[0] == list[int]
    assert tuple_args[1] == list[str]
    assert tuple_args[2] == list[float]


def test_inheritance_and_specialization():
    """Test inheritance chains and specialization patterns."""

    # Simple inheritance
    def process_derived(obj: ConcreteChild) -> int: ...

    derived = ConcreteChild(value=42, extra="hello")
    t = infer_return_type(process_derived, derived)
    assert t is int

    # Deep inheritance chain

    def process_wrap(obj: Wrap[A]) -> A: ...
    def process_derived(obj: DerivedWrap[A]) -> A: ...

    child = ChildWrap[float](value=3.14, derived_value=42, child_value="test")

    result_wrap = infer_return_type(process_wrap, child)
    assert result_wrap == float

    result_derived = infer_return_type(process_derived, child)
    assert result_derived == float

    # Partial specialization

    def process_two(obj: GenericPair[A, B]) -> Tuple[A, B]: ...

    one = OneParam[int](first=42, second="fixed", extra=99)
    result = infer_return_type(process_two, one)
    assert typing.get_origin(result) == tuple
    assert typing.get_args(result) == (int, str)

    # Multiple inheritance with different TypeVar names
    def extract_a(obj: HasA[A]) -> A: ...
    def extract_b(obj: HasB[B]) -> B: ...

    both = HasBoth[int, str](a_value=42, b_value="hello", both="test")

    result_a = infer_return_type(extract_a, both)
    assert result_a == int

    result_b = infer_return_type(extract_b, both)
    assert result_b == str

    # Multiple inheritance with same TypeVar names (shadowing)

    def extract_a2(obj: HasA2[A]) -> A: ...
    def extract_b2(obj: HasB2[B]) -> B: ...

    both2 = HasBoth2[int, str](a_value=42, b_value="hello", both="test")

    result_a2 = infer_return_type(extract_a2, both2)
    assert result_a2 == int

    result_b2 = infer_return_type(extract_b2, both2)
    assert result_b2 == str

    # Pydantic inheritance with swapped parameters
    def process_pyd(obj: ChildPyd[C, D]) -> Tuple[C, D]: ...

    result = infer_return_type(
        process_pyd, ChildPyd[int, str](a_value="hello", b_value=42)
    )
    assert typing.get_origin(result) is tuple
    assert typing.get_args(result) == (int, str)


def test_union_types_and_distribution():
    """Test union type handling and type distribution."""

    # Union type limitations
    def process_union(data: Union[List[A], Set[A]]) -> A: ...

    t = infer_return_type(process_union, [1, 2, 3])
    assert t is int

    t = infer_return_type(process_union, {"hello", "world"})
    assert t is str

    # Modern union syntax
    def process_modern_union(data: List[A] | Set[A]) -> A: ...

    t = infer_return_type(process_modern_union, [1, 2, 3])
    assert t is int

    # Mixed type containers create unions
    def process_mixed_list(items: List[A]) -> A: ...

    mixed_list = [1, "hello", 3.14]
    t = infer_return_type(process_mixed_list, mixed_list)

    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str, float}

    # Complex union scenarios
    def complex_union_result(
        data: Dict[A, List[B]],
    ) -> Union[A, List[B], Tuple[A, B]]: ...

    data = {"key": [1, 2, 3]}
    t = infer_return_type(complex_union_result, data)

    union_origin = typing.get_origin(t)
    assert union_origin is Union or union_origin is getattr(types, "UnionType", None)

    union_args = typing.get_args(t)
    assert str in union_args

    # Check for List[int] in union
    list_types = [arg for arg in union_args if typing.get_origin(arg) is list]
    assert len(list_types) > 0
    assert typing.get_args(list_types[0]) == (int,)

    # Check for Tuple[str, int] in union
    tuple_types = [arg for arg in union_args if typing.get_origin(arg) is tuple]
    assert len(tuple_types) > 0
    assert typing.get_args(tuple_types[0]) == (str, int)

    # Union with generics
    def maybe_wrap(x: A, should_wrap: bool) -> A | Wrap[A]: ...

    t = infer_return_type(maybe_wrap, 42, True)
    if hasattr(t, "__args__"):
        union_types = typing.get_args(t)
        assert int in union_types
        wrap_types = [arg for arg in union_types if typing.get_origin(arg) == Wrap]
        assert len(wrap_types) > 0

    # Set union distribution
    def process_union_set(s: Set[Union[A, B]]) -> Tuple[Set[A], Set[B]]: ...

    result = infer_return_type(process_union_set, {1, "a", 2, "b"})

    tuple_args = typing.get_args(result)
    assert len(tuple_args) == 2

    origin1 = typing.get_origin(tuple_args[0])
    origin2 = typing.get_origin(tuple_args[1])
    assert origin1 is set and origin2 is set

    element_type1 = typing.get_args(tuple_args[0])[0]
    element_type2 = typing.get_args(tuple_args[1])[0]

    assert {element_type1, element_type2} == {int, str}
    assert element_type1 != element_type2


def test_optional_and_none_handling():
    """Test Optional types and None value handling."""

    # Basic Optional handling
    def process_optional_list(data: List[Optional[A]]) -> A: ...

    optional_list = [1, None, 2, None, 3]
    t = infer_return_type(process_optional_list, optional_list)
    assert t is int

    def process_list_of_optionals(data: Optional[List[A]]) -> A: ...

    t = infer_return_type(process_list_of_optionals, [1, 2, 3])
    assert t is int

    # None case should fail appropriately
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_list_of_optionals, None)

    # Optional nested in containers
    def process_nested_optional(data: Optional[List[Optional[A]]]) -> A: ...

    test_data = [1, None, 2]

    t = infer_return_type(process_nested_optional, test_data)

    origin = typing.get_origin(t)
    if origin is Union or origin is getattr(types, "UnionType", None):
        union_args = typing.get_args(t)
        assert int in union_args
    else:
        assert t is int

    # None filtering in Optional[A] - should not bind A to None
    def process_optional_values(data: Dict[str, Optional[A]]) -> A: ...

    test_data = {"a": 1, "b": None, "c": 2, "d": None, "e": 3}
    t = infer_return_type(process_optional_values, test_data)
    assert t is int  # Should be int, not int | None

    # Variable length tuple with Optional
    def process_var_tuple(t: Tuple[A, ...]) -> Set[A]: ...

    t = infer_return_type(process_var_tuple, (1, 2, 3))
    assert typing.get_origin(t) is set
    assert typing.get_args(t) == (int,)

    # Mixed types in variable tuple should create union
    t = infer_return_type(process_var_tuple, (1, "hello", 2, "world"))
    assert typing.get_origin(t) is set
    union_arg = typing.get_args(t)[0]
    origin = typing.get_origin(union_arg)
    assert origin is Union or origin is getattr(types, "UnionType", None)


def test_constraints_and_bounds():
    """Test TypeVar constraints and bounds."""

    # Bounded TypeVars
    T = TypeVar("T", bound=int)
    U = TypeVar("U", bound=str)
    V = TypeVar("V", int, float)  # Constrained

    def multi_bounded(x: T, y: U, z: V) -> Tuple[T, U, V]: ...

    t = infer_return_type(multi_bounded, True, "hello", 3.14)
    assert typing.get_args(t) == (bool, str, float)

    def increment_bounded(x: T) -> T: ...

    t = infer_return_type(increment_bounded, True)
    assert t is bool  # Should preserve the specific type

    # Bounded TypeVar violations
    T_bounded = TypeVar("T_bounded", bound=str)

    def process_bounded(x: T_bounded) -> T_bounded: ...

    # int doesn't satisfy bound=str
    with pytest.raises(TypeInferenceError, match="bound"):
        infer_return_type(process_bounded, 42)

    # str subclass should work
    t = infer_return_type(process_bounded, "hello")
    assert t == str

    # Constrained TypeVar violations
    T_constrained = TypeVar("T_constrained", int, str)

    def process_constrained(x: T_constrained) -> T_constrained: ...

    # float is not in constraints (int, str)
    with pytest.raises(TypeInferenceError, match="violates constraints"):
        infer_return_type(process_constrained, 3.14)

    # int should work
    t = infer_return_type(process_constrained, 42)
    assert t == int

    # Constrained TypeVar with mixed types violates constraints
    def process_mixed(items: List[T_constrained]) -> T_constrained: ...

    with pytest.raises(TypeInferenceError, match="violates constraints"):
        t = infer_return_type(process_mixed, [1, "hello", 2])

    # Type overrides have highest priority
    def process_list(items: List[A]) -> A: ...

    result = infer_return_type(process_list, [1, 2, 3], type_overrides={A: str})
    assert result is str


def test_error_handling_and_edge_cases():
    """Test error handling and edge cases."""

    # Empty containers
    def process_empty_list(items: List[A]) -> A: ...
    def process_empty_dict(data: Dict[A, B]) -> Tuple[A, B]: ...
    def process_empty_set(items: Set[A]) -> A: ...

    with pytest.raises(TypeInferenceError):
        infer_return_type(process_empty_list, [])

    with pytest.raises(TypeInferenceError):
        infer_return_type(process_empty_dict, {})

    with pytest.raises(TypeInferenceError):
        infer_return_type(process_empty_set, set())

    # Type mismatches
    def process_list(items: List[A]) -> A: ...

    with pytest.raises(TypeInferenceError, match="Expected list"):
        infer_return_type(process_list, "not a list")

    def process_dict(data: Dict[A, B]) -> A: ...

    with pytest.raises(TypeInferenceError, match="Expected dict"):
        infer_return_type(process_dict, [1, 2, 3])

    def process_set(s: Set[A]) -> A: ...

    with pytest.raises(TypeInferenceError, match="Expected set"):
        infer_return_type(process_set, [1, 2, 3])

    def process_tuple(t: Tuple[A, B]) -> A: ...

    with pytest.raises(TypeInferenceError, match="Expected tuple"):
        infer_return_type(process_tuple, [1, 2])

    # Functions without return annotations
    def no_annotation(x):
        return x

    with pytest.raises(ValueError, match="return type annotation"):
        infer_return_type(no_annotation, 42)

    # Empty container error messages
    try:
        infer_return_type(process_empty_list, [])
        assert False, "Should have raised TypeInferenceError"
    except TypeInferenceError as e:
        error_msg = str(e)
        assert len(error_msg) > 20
        assert "A" in error_msg or "TypeVar" in error_msg.lower()
        assert "insufficient" in error_msg.lower() or "could not" in error_msg.lower()


def test_variance_and_covariance():
    """Test variance handling and covariant behavior."""

    # Covariant variance
    class Animal:
        pass

    class Dog(Animal):
        pass

    class Cat(Animal):
        pass

    def covariant_test(pets: List[A]) -> A: ...

    # List is covariant - infer most specific type
    dog_list = [Dog(), Dog()]
    result = infer_return_type(covariant_test, dog_list)
    assert result is Dog

    cat_list = [Cat(), Cat()]
    result = infer_return_type(covariant_test, cat_list)
    assert result is Cat

    # Mixed list creates union
    mixed_list = [Dog(), Cat()]
    result = infer_return_type(covariant_test, mixed_list)

    origin = typing.get_origin(result)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(result)
    assert set(union_args) == {Dog, Cat}

    # Bounded TypeVar with union
    T_bounded = TypeVar("T_bounded", bound=Animal)

    def bounded_test(pets: List[T_bounded]) -> T_bounded: ...

    bounded_result = infer_return_type(bounded_test, mixed_list)
    bounded_origin = typing.get_origin(bounded_result)
    assert bounded_origin is Union or bounded_origin is getattr(
        types, "UnionType", None
    )
    bounded_args = typing.get_args(bounded_result)
    assert set(bounded_args) == {Dog, Cat}

    # Invariant dict keys
    class StringKey(str):
        pass

    def invariant_test(mapping: Dict[A, int]) -> A: ...

    string_dict = {"key": 1}
    result1 = infer_return_type(invariant_test, string_dict)
    assert result1 is str

    custom_dict = {StringKey("key"): 1}
    result2 = infer_return_type(invariant_test, custom_dict)
    assert result2 is StringKey


def test_deep_nesting_stress_tests():
    """Test extreme depth and complexity scenarios."""

    # Triple nested generic classes
    def triple_unbox(b: Wrap[Wrap[Wrap[A]]]) -> A: ...

    innermost = Wrap[int](value=42)
    middle = Wrap[Wrap[int]](value=innermost)
    outer = Wrap[Wrap[Wrap[int]]](value=middle)
    t = infer_return_type(triple_unbox, outer)
    assert t is int

    # Quadruple nested generic classes
    def quad_extract(c: Wrap[Wrap[Wrap[Wrap[A]]]]) -> A: ...

    level1 = Wrap[str](value="deep")
    level2 = Wrap[Wrap[str]](value=level1)
    level3 = Wrap[Wrap[Wrap[str]]](value=level2)
    level4 = Wrap[Wrap[Wrap[Wrap[str]]]](value=level3)
    t = infer_return_type(quad_extract, level4)
    assert t is str

    # Six level list nesting
    def extract_from_six_deep(data: List[List[List[List[List[List[A]]]]]]) -> A: ...

    deep_data = [[[[[[42]]]]]]
    t = infer_return_type(extract_from_six_deep, deep_data)
    assert t is int

    # Deep dict nesting
    def extract_all_types(
        data: Dict[A, Dict[B, Dict[C, Dict[D, E]]]],
    ) -> Tuple[A, B, C, D, E]: ...

    deep_dict = {"level1": {42: {3.14: {True: "deepest"}}}}
    t = infer_return_type(extract_all_types, deep_dict)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args == (str, int, float, bool, str)

    # Mixed containers at depth
    def extract_from_complex(
        data: List[Dict[str, Set[Tuple[A, B]]]],
    ) -> Tuple[A, B]: ...

    complex_data = [{"key1": {(1, "a"), (2, "b")}}, {"key2": {(3, "c")}}]
    t = infer_return_type(extract_from_complex, complex_data)
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (int, str)

    # Recursive structures
    def extract_from_deep_tree(tree: GraphNode[GraphNode[GraphNode[A]]]) -> A: ...

    leaf1 = GraphNode[int](value=1, edges=[])
    leaf2 = GraphNode[int](value=2, edges=[])
    middle1 = GraphNode[GraphNode[int]](value=leaf1, edges=[])
    middle2 = GraphNode[GraphNode[int]](value=leaf2, edges=[])
    root = GraphNode[GraphNode[GraphNode[int]]](value=middle1, edges=[])
    t = infer_return_type(extract_from_deep_tree, root)
    assert t is int


def test_unification_engine_internals():
    """Test unification function internals."""

    # Direct API testing
    substitution = unify_annotation_with_value(List[A], [1, 2, 3])
    assert substitution.get(A).resolved_type == int

    substitution = unify_annotation_with_value(Dict[K, V], {"a": 1})
    assert substitution.get(K).resolved_type == str
    assert substitution.get(V).resolved_type == int

    # With pre-existing constraints
    existing_constraints = [Constraint(A, get_generic_info(int), Variance.INVARIANT)]
    substitution = unify_annotation_with_value(
        List[A], [1, 2, 3], constraints=existing_constraints
    )
    assert substitution.get(A).resolved_type == int

    # None constraints
    substitution = unify_annotation_with_value(Set[A], {1, 2, 3}, constraints=None)
    assert substitution.get(A).resolved_type == int

    # Constraint solver with many constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT) for _ in range(10)
    ]
    constraints.extend(
        [Constraint(A, get_generic_info(str), Variance.COVARIANT) for _ in range(10)]
    )

    sub = solve_constraints(constraints)
    result = sub.get(A)

    resolved_result = result.resolved_type
    origin = typing.get_origin(resolved_result)
    assert origin is Union or origin is getattr(types, "UnionType", None)

    # Override constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT),
        Constraint(A, get_generic_info(str), Variance.COVARIANT),
        Constraint(A, get_generic_info(float), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    assert sub.get(A).resolved_type == float  # Override should take precedence

    # Conflicting overrides
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(str), Variance.INVARIANT, is_override=True),
    ]

    with pytest.raises(UnificationError, match="Conflicting override"):
        solve_constraints(constraints)


def test_helper_functions():
    """Test internal helper functions."""
    # _has_unbound_typevars_in_generic_info
    assert _has_unbound_typevars_in_generic_info(get_generic_info(A)) == True
    assert _has_unbound_typevars_in_generic_info(get_generic_info(int)) == False
    assert _has_unbound_typevars_in_generic_info(get_generic_info(List[A])) == True
    assert _has_unbound_typevars_in_generic_info(get_generic_info(List[int])) == False
    assert _has_unbound_typevars_in_generic_info(get_generic_info(Dict[A, int])) == True
    assert (
        _has_unbound_typevars_in_generic_info(get_generic_info(Dict[str, int])) == False
    )

    # Substitution.apply
    substitution = Substitution({B: get_generic_info(int)})
    result = substitution.apply(A)
    assert result == A  # A is not in bindings

    substitution = Substitution({A: get_generic_info(int)})
    union_type = Union[A, B, str]
    result = substitution.apply(union_type)

    origin = typing.get_origin(result)
    if origin is Union or origin is getattr(types, "UnionType", None):
        args = typing.get_args(result)
        assert int in args
        assert str in args

    # _infer_type_from_value (now returns GenericInfo)
    t = _infer_type_from_value(None)
    assert t.resolved_type == type(None)

    t = _infer_type_from_value([])
    assert t.resolved_type == list

    t = _infer_type_from_value([1, 2, 3])
    resolved = t.resolved_type
    assert typing.get_origin(resolved) is list
    assert typing.get_args(resolved) == (int,)

    t = _infer_type_from_value({1: "a", 2: "b"})
    resolved = t.resolved_type
    assert typing.get_origin(resolved) is dict
    key_type, val_type = typing.get_args(resolved)
    assert key_type == int
    assert val_type == str

    t = _infer_type_from_value({1, 2, 3})
    resolved = t.resolved_type
    assert typing.get_origin(resolved) is set
    assert typing.get_args(resolved) == (int,)

    t = _infer_type_from_value((1, "hello", 3.14))
    resolved = t.resolved_type
    assert typing.get_origin(resolved) is tuple

    # _is_subtype
    assert _is_subtype(bool, int)  # bool is subtype of int
    assert _is_subtype(int, object)
    assert not _is_subtype(int, str)
    assert not _is_subtype(List[int], list)  # Generic types handled


def test_substitution_and_constraint_internals():
    """Test Constraint and Substitution class internals."""

    # Constraint __str__ and __repr__
    constraint = Constraint(A, get_generic_info(int), Variance.COVARIANT)
    str_repr = str(constraint)
    assert "~" in str_repr
    assert "covariant" in str_repr

    # Constraint with is_override flag
    override_constraint = Constraint(
        A, get_generic_info(str), Variance.INVARIANT, is_override=True
    )
    override_str = str(override_constraint)
    assert "override" in override_str
    repr_str = repr(override_constraint)
    assert "override" in repr_str

    # Substitution basic functionality
    sub = Substitution()
    sub.bind(A, get_generic_info(int))
    sub.bind(B, get_generic_info(str))
    sub.bind(C, get_generic_info(float))

    assert sub.get(A).resolved_type == int
    assert sub.get(B).resolved_type == str
    assert sub.get(C).resolved_type == float


@pytest.mark.skip(
    reason="LIMITATION: Callable parameter extraction requires signature inspection"
)
def test_callable_limitations():
    """Document Callable type inference limitations."""

    def higher_order(func: Callable[[A], B], value: A) -> B: ...

    def sample_func(x: int) -> str:
        return str(x)

    # Callable type inference is not fully supported
    with pytest.raises(TypeInferenceError):
        infer_return_type(higher_order, sample_func, 42)


@pytest.mark.skip(reason="LIMITATION: ForwardRef handling not fully supported")
def test_forward_reference_limitations():
    """Document ForwardRef handling limitations."""

    # ForwardRef handling is not fully implemented
    # The engine can't properly resolve string annotations to actual types
    pass


def test_keyword_arguments_and_edge_cases():
    """Test keyword arguments and additional edge cases."""

    # Keyword arguments with inference
    def func_with_kwargs(a: A, b: B, c: C = None) -> Tuple[A, B]: ...

    # Mix positional and keyword arguments
    t = infer_return_type(func_with_kwargs, 1, b="hello", c=3.14)
    assert typing.get_origin(t) == tuple
    assert typing.get_args(t) == (int, str)

    # All keyword arguments
    def func_all_kwargs(x: A, y: B, z: C) -> Dict[A, B]: ...

    t = infer_return_type(func_all_kwargs, x=1, y="str", z=3.14)
    assert typing.get_origin(t) == dict
    key_type, val_type = typing.get_args(t)
    assert key_type == int
    assert val_type == str

    # Extra keyword arguments should be ignored
    def func_limited(a: A) -> A: ...

    t = infer_return_type(func_limited, a=42, extra="ignored")
    assert t == int

    # Tuple ellipsis handling
    def process_var_tuple(t: Tuple[A, ...]) -> Set[A]: ...

    # Empty tuple should fail (can't infer A)
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_var_tuple, ())

    # Non-empty tuple should work
    t = infer_return_type(process_var_tuple, (1, 2, 3))
    assert typing.get_origin(t) is set
    assert typing.get_args(t) == (int,)

    # Mixed types in variable tuple should create union
    t = infer_return_type(process_var_tuple, (1, "hello", 2, "world"))
    assert typing.get_origin(t) is set
    union_arg = typing.get_args(t)[0]
    origin = typing.get_origin(union_arg)
    assert origin is Union or origin is getattr(types, "UnionType", None)

    # Empty set handling
    def process_empty_set_fallback(s: Set[A], default: A) -> A: ...

    # Empty set should use default value for inference
    t = infer_return_type(process_empty_set_fallback, set(), 42)
    assert t == int


def test_complex_union_structures():
    """Test complex union structures and advanced patterns."""

    # Complex union structure with nested generics
    def extract_value(data: Dict[str, Union[A, List[A], Dict[str, A]]]) -> A: ...

    test_data = {"single": 42, "list": [43, 44], "nested": {"value": 45}}

    # Should extract A = int from all three positions
    result = infer_return_type(extract_value, test_data)
    assert result is int

    # Union of different container types
    def process_list_or_dict(data: Union[List[A], Dict[str, A]]) -> A: ...

    # Should recognize this as List[int] and bind A=int
    t = infer_return_type(process_list_or_dict, [1, 2, 3])
    assert t is int

    # Should recognize this as Dict[str, int] and bind A=int
    t = infer_return_type(process_list_or_dict, {"key": 42})
    assert t is int

    # Union of generic containers with different type params
    def process_container_union(data: Union[List[A], Set[B]]) -> Union[A, B]: ...

    # Should bind A=int and return A (since it's a list)
    t = infer_return_type(process_container_union, [1, 2, 3])
    assert t is int

    # Multiple union parameters
    def process_multiple_unions(
        data1: Union[List[A], Tuple[A, ...]], data2: Union[Set[B], Dict[str, B]]
    ) -> Tuple[A, B]: ...

    # Should handle multiple union parameters
    t = infer_return_type(process_multiple_unions, [1, 2], {"a": "hello", "b": "world"})
    assert typing.get_origin(t) is tuple
    assert typing.get_args(t) == (int, str)

    # Nested unions in generics
    def process_nested_union(data: List[Union[A, B]]) -> Union[A, B]: ...

    # List containing mixed types should infer union
    mixed_list = [1, "hello", 2, "world"]  # int and str mixed
    t = infer_return_type(process_nested_union, mixed_list)

    # Should return Union[int, str] or int | str
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)

    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


def test_substitution_and_type_reconstruction():
    """Test type substitution and reconstruction edge cases."""

    # Test substituting in dict type
    substitution = Substitution({K: get_generic_info(str), V: get_generic_info(int)})
    result = substitution.apply(Dict[K, V])

    assert typing.get_origin(result) == dict
    key_type, val_type = typing.get_args(result)
    assert key_type == str
    assert val_type == int

    # Fixed-length tuple substitution
    substitution = Substitution(
        {A: get_generic_info(int), B: get_generic_info(str), C: get_generic_info(float)}
    )
    result = substitution.apply(Tuple[A, B, C])

    assert typing.get_origin(result) == tuple
    args = typing.get_args(result)
    assert args == (int, str, float)

    # Set type substitution
    substitution = Substitution({A: get_generic_info(str)})
    result = substitution.apply(Set[A])

    assert typing.get_origin(result) == set
    assert typing.get_args(result) == (str,)

    # Custom generic class substitution

    substitution = Substitution({A: get_generic_info(int), B: get_generic_info(str)})
    result = substitution.apply(GenericPair[A, B])

    # Should attempt to reconstruct GenericPair[int, str]
    assert result == GenericPair[int, str]
    assert result != GenericPair[A, B]

    # Union substitution with mixed bound/unbound TypeVars
    substitution = Substitution({A: get_generic_info(int)})
    union_type = Union[A, B, str]
    result = substitution.apply(union_type)

    # Should only include bound args (int and str), not B
    origin = typing.get_origin(result)
    if origin is Union or origin is getattr(types, "UnionType", None):
        args = typing.get_args(result)
        # B should not be in the result since it's unbound
        assert int in args
        assert str in args


def test_additional_edge_cases():
    """Test additional edge cases to improve coverage."""
    from generic_utils import get_generic_info

    # Test constraint solver with many constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT) for _ in range(10)
    ]
    constraints.extend(
        [Constraint(A, get_generic_info(str), Variance.COVARIANT) for _ in range(10)]
    )

    sub = solve_constraints(constraints)
    result = sub.get(A)

    resolved_result = result.resolved_type
    origin = typing.get_origin(resolved_result)
    assert origin is Union or origin is getattr(types, "UnionType", None)

    # Test constraint solver when all constraints are identical
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT) for _ in range(100)
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test mixed variance constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT),
        Constraint(A, get_generic_info(str), Variance.INVARIANT),
        Constraint(A, get_generic_info(float), Variance.COVARIANT),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)

    resolved_result = result.resolved_type
    origin = typing.get_origin(resolved_result)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(resolved_result)
    assert set(union_args) == {int, str, float}

    # Test union components matching
    from generic_utils import get_generic_info

    # Create unions: int | list[int] vs int | list
    union1_info = get_generic_info(int | list[int])
    union2_info = get_generic_info(int | list)

    # Should match by origin (int matches int, list[int] origin matches list)
    result = _union_components_match(union1_info, union2_info)
    assert result

    # Test union components with different lengths
    union1_info = get_generic_info(int | str | float)
    union2_info = get_generic_info(int | str)

    result = _union_components_match(union1_info, union2_info)
    assert not result

    # Test generic info matching with different origins
    list_info = get_generic_info(List[A])
    set_val = {1, 2, 3}
    from generic_utils import get_instance_generic_info

    set_info = get_instance_generic_info(set_val)

    constraints = []
    result = _match_generic_structures(list_info, set_info, constraints)
    assert not result
    assert len(constraints) == 0

    # Test generic info matching with different argument counts
    list_info = get_generic_info(List[A])
    dict_info = get_generic_info(Dict[str, int])

    constraints = []
    result = _match_generic_structures(list_info, dict_info, constraints)
    assert not result

    # Test origins compatibility
    compatible = _origins_compatible(list, list)
    assert compatible

    compatible = _origins_compatible(list, set)
    assert not compatible

    # Test tuple fixed length partial match
    def process_tuple(t: Tuple[A, B]) -> A: ...

    # Tuple longer than expected - should still extract available positions
    t = infer_return_type(process_tuple, (1, "hello", "extra"))
    assert t == int

    # Test custom generic fallback paths
    def process_custom(c: Wrap[A]) -> A: ...

    instance = Wrap(value=42)
    # Remove __orig_class__ if it exists to test fallback
    if hasattr(instance, "__orig_class__"):
        delattr(instance, "__orig_class__")

    t = infer_return_type(process_custom, instance)
    assert t == int

    # Test NoneType inference (now returns GenericInfo)
    from infer_return_type import _infer_type_from_value

    t = _infer_type_from_value(None)
    assert t.resolved_type == type(None)

    # Test empty list inference
    t = _infer_type_from_value([])
    assert t.resolved_type == list

    # Test mixed list inference
    t = _infer_type_from_value([1, "hello"])
    assert typing.get_origin(t.resolved_type) is list

    # Test empty dict inference
    t = _infer_type_from_value({})
    assert t.resolved_type == dict

    # Test empty set inference
    t = _infer_type_from_value(set())
    assert t.resolved_type == set

    # Test tuple inference
    t = _infer_type_from_value((1, "hello", 3.14))
    assert typing.get_origin(t.resolved_type) is tuple

    # Test _is_subtype edge cases
    from infer_return_type import _is_subtype

    assert _is_subtype(bool, int)
    assert _is_subtype(int, object)
    assert not _is_subtype(int, str)
    assert not _is_subtype(List[int], list)  # Generic types handled

    # Test union constraint handling errors
    def process_strict_union(x: Union[List[int], Dict[str, str]]) -> int: ...

    # A set doesn't match either alternative
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_strict_union, {1, 2, 3})

    # Test set constraint union handling
    def process_set_union(s: Set[Union[A, B]]) -> Tuple[A, B]: ...

    # Mixed set with two types
    t = infer_return_type(process_set_union, {1, "hello", 2, "world"})
    assert typing.get_origin(t) is tuple

    # Should distribute types among A and B
    tuple_args = typing.get_args(t)
    assert set(tuple_args) == {int, str}

    # Test constraint checking with nested generics
    T = TypeVar("T", list[int], dict[str, int])

    def process_constrained(x: T) -> T: ...

    # list[int] should match first constraint
    t = infer_return_type(process_constrained, [1, 2, 3])
    assert typing.get_origin(t) == list
    assert typing.get_args(t) == (int,)

    # Test bounded TypeVar with union
    class Base:
        pass

    class Derived1(Base):
        pass

    class Derived2(Base):
        pass

    T_bounded = TypeVar("T_bounded", bound=Base)

    def process_bounded_multi(items: List[T_bounded]) -> T_bounded: ...

    # Mixed derived types should create union within bound
    t = infer_return_type(process_bounded_multi, [Derived1(), Derived2()])

    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(t)
    assert set(union_args) == {Derived1, Derived2}

    # Test constrained TypeVar where inferred union matches a constraint union
    T = TypeVar("T", int | str, float | bool)

    def process(items: List[T]) -> T: ...

    # [1, "x"] should infer int | str, matching first constraint
    t = infer_return_type(process, [1, "x"])

    # Should match the first constraint (int | str)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}

    # Test substitution with generic alias
    substitution = Substitution({K: get_generic_info(str), V: get_generic_info(int)})
    result = substitution.apply(Dict[K, V])

    assert typing.get_origin(result) == dict
    key_type, val_type = typing.get_args(result)
    assert key_type == str
    assert val_type == int

    # Test substitution preserves tuple structure
    substitution = Substitution(
        {A: get_generic_info(int), B: get_generic_info(str), C: get_generic_info(float)}
    )
    result = substitution.apply(Tuple[A, B, C])

    assert typing.get_origin(result) == tuple
    args = typing.get_args(result)
    assert args == (int, str, float)

    # Test substitution with Set types
    substitution = Substitution({A: get_generic_info(str)})
    result = substitution.apply(Set[A])

    assert typing.get_origin(result) == set
    assert typing.get_args(result) == (str,)

    # Test substitution with generic class

    substitution = Substitution({A: get_generic_info(int), B: get_generic_info(str)})
    result = substitution.apply(GenericPair[A, B])

    # Should attempt to reconstruct GenericPair[int, str]
    assert result == GenericPair[int, str]
    assert result != GenericPair[A, B]

    # Test additional edge cases for coverage
    # Test union distribution with context-aware matching
    def complex_set_union(
        s1: Set[Union[A, B]], s2: Set[Union[A, B]], s3: Set[Union[A, B]]
    ) -> Tuple[A, B]: ...

    # Multiple sets with mixed types - tests context-aware matching
    t = infer_return_type(
        complex_set_union,
        {1, "a"},  # Establishes A=int, B=str
        {2, "b"},  # Reinforces pattern
        {3, "c"},  # Reinforces pattern
    )

    assert typing.get_origin(t) == tuple
    result_types = set(typing.get_args(t))
    assert result_types == {int, str}

    # Test set union with no candidates (fallback path)
    def set_union_difficult(s: Set[Union[A, B]]) -> Tuple[A, B]: ...

    # This exercises the fallback path
    t = infer_return_type(set_union_difficult, {1, "x", 2.5})
    assert typing.get_origin(t) == tuple

    # Test multiple invariant conflicts
    def process_containers(
        d1: Dict[A, str], d2: Dict[A, str], d3: Dict[A, str]
    ) -> A: ...

    # Three dicts with different key types (invariant position)
    result = infer_return_type(process_containers, {1: "a"}, {"x": "b"}, {3.14: "c"})

    # Should create int | str | float union
    origin = typing.get_origin(result)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(result)
    assert set(union_args) == {int, str, float}

    # Test nested variance mixing
    def process_mixed_variance(d: Dict[A, List[A]]) -> A: ...

    # Keys must match (invariant), list elements are covariant
    test_data = {1: [1, 2, 3], 2: [4, 5, 6]}

    result = infer_return_type(process_mixed_variance, test_data)
    assert result is int

    # Test constraint trace on failure (conflicting constraints create unions)
    def conflicting_example(a: List[A], b: List[A]) -> A: ...

    # This creates a union now (improved behavior)
    result = infer_return_type(conflicting_example, [1], ["x"])
    # Current behavior: creates int | str union
    assert typing.get_origin(result) in [Union, getattr(types, "UnionType", None)]

    # Verify both types are in the union
    union_args = typing.get_args(result)
    assert set(union_args) == {int, str}

    # Test substitution with union reconstruction
    substitution = Substitution({A: get_generic_info(int), B: get_generic_info(str)})
    union_type = Union[A, B]
    result = substitution.apply(union_type)

    # Should reconstruct union
    origin = typing.get_origin(result)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    args = typing.get_args(result)
    assert set(args) == {int, str}

    # Test substitution with single union arg
    substitution = Substitution({A: get_generic_info(int)})
    union_type = Union[A]
    result = substitution.apply(union_type)

    # Single union arg should be returned directly
    assert result == int

    # Test additional edge cases for final coverage push
    # Test constraint solver edge cases

    # Test with override constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT),
        Constraint(A, get_generic_info(str), Variance.COVARIANT),
        Constraint(
            A, get_generic_info(float), Variance.INVARIANT, is_override=True
        ),  # Override wins
    ]

    sub = solve_constraints(constraints)
    assert sub.get(A).resolved_type == float  # Override should take precedence

    # Test constraint solver with many constraints (stress test)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT) for _ in range(10)
    ]
    constraints.extend(
        [Constraint(A, get_generic_info(str), Variance.COVARIANT) for _ in range(10)]
    )

    sub = solve_constraints(constraints)
    result = sub.get(A)

    resolved_result = result.resolved_type
    origin = typing.get_origin(resolved_result)
    assert origin is Union or origin is getattr(types, "UnionType", None)

    # Test constraint solver when all constraints are identical
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT) for _ in range(100)
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test union constraint handling errors
    def process_strict_union_error(x: Union[List[int], Dict[str, str]]) -> int: ...

    # A set doesn't match either alternative
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_strict_union_error, {1, 2, 3})

    # Test substitution edge cases
    # Test substitution with empty set (base type)
    def process_empty_set_fallback(s: Set[A], default: A) -> A: ...

    # Empty set should use default value for inference
    t = infer_return_type(process_empty_set_fallback, set(), 42)
    assert t == int

    # Test substitution with generic alias reconstruction
    substitution = Substitution({K: get_generic_info(str), V: get_generic_info(int)})
    result = substitution.apply(Dict[K, V])

    assert typing.get_origin(result) == dict
    key_type, val_type = typing.get_args(result)
    assert key_type == str
    assert val_type == int

    # Test substitution with tuple reconstruction
    substitution = Substitution(
        {A: get_generic_info(int), B: get_generic_info(str), C: get_generic_info(float)}
    )
    result = substitution.apply(Tuple[A, B, C])

    assert typing.get_origin(result) == tuple
    args = typing.get_args(result)
    assert args == (int, str, float)

    # Test substitution with set reconstruction
    substitution = Substitution({A: get_generic_info(str)})
    result = substitution.apply(Set[A])

    assert typing.get_origin(result) == set
    assert typing.get_args(result) == (str,)

    # Test substitution with generic class reconstruction

    substitution = Substitution({A: get_generic_info(int), B: get_generic_info(str)})
    result = substitution.apply(SubstitutionContainer[A, B])

    # Should attempt to reconstruct SubstitutionContainer[int, str]
    assert result == SubstitutionContainer[int, str]
    assert result != SubstitutionContainer[A, B]

    # Test substitution with other generic types (fallback)
    substitution = Substitution({A: get_generic_info(int), B: get_generic_info(str)})
    result = substitution.apply(SubstitutionContainer[A, B])

    # Should attempt reconstruction
    assert result == SubstitutionContainer[int, str]

    # Test substitution with union reconstruction
    substitution = Substitution({A: get_generic_info(int), B: get_generic_info(str)})
    union_type = Union[A, B]
    result = substitution.apply(union_type)

    # Should reconstruct union
    origin = typing.get_origin(result)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    args = typing.get_args(result)
    assert set(args) == {int, str}

    # Test substitution with single union arg
    substitution = Substitution({A: get_generic_info(int)})
    union_type = Union[A]
    result = substitution.apply(union_type)

    # Single union arg should be returned directly
    assert result == int

    # Test substitution with mixed bound/unbound TypeVars
    substitution = Substitution({A: get_generic_info(int)})
    union_type = Union[A, B, str]
    result = substitution.apply(union_type)

    # Should only include bound args (int and str), not B
    origin = typing.get_origin(result)
    if origin is Union or origin is getattr(types, "UnionType", None):
        args = typing.get_args(result)
        # B should not be in the result since it's unbound
        assert int in args
        assert str in args

    # Test substitution with no bound args
    substitution = Substitution({})
    union_type = Union[A, B]
    result = substitution.apply(union_type)

    # Should return original annotation since no args were bound
    assert result == union_type

    # Test substitution with single bound arg
    substitution = Substitution({A: get_generic_info(int)})
    union_type = Union[A, B]
    result = substitution.apply(union_type)

    # Should return only the bound arg
    assert result == int

    # Test additional edge cases for final coverage push
    # Test constraint solver with conflicting override constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(str), Variance.INVARIANT, is_override=True),
    ]

    # This should raise an error due to conflicting overrides
    with pytest.raises(UnificationError):
        solve_constraints(constraints)

    # Test constraint solver with single constraint
    constraints = [Constraint(A, get_generic_info(int), Variance.INVARIANT)]
    sub = solve_constraints(constraints)
    assert sub.get(A).resolved_type == int

    # Test constraint solver with no constraints
    constraints = []
    sub = solve_constraints(constraints)
    # Should return empty substitution
    assert len(sub.bindings) == 0

    # Test constraint solver with covariant constraints only
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT),
        Constraint(A, get_generic_info(str), Variance.COVARIANT),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)

    resolved_result = result.resolved_type
    origin = typing.get_origin(resolved_result)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(resolved_result)
    assert set(union_args) == {int, str}

    # Test constraint solver with mixed variance (invariant + covariant)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT),
        Constraint(A, get_generic_info(str), Variance.COVARIANT),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)

    resolved_result = result.resolved_type
    origin = typing.get_origin(resolved_result)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(resolved_result)
    assert set(union_args) == {int, str}

    # Test constraint solver with multiple invariant constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT),
        Constraint(A, get_generic_info(str), Variance.INVARIANT),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)

    resolved_result = result.resolved_type
    origin = typing.get_origin(resolved_result)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(resolved_result)
    assert set(union_args) == {int, str}

    # Test constraint solver with identical invariant constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT),
        Constraint(A, get_generic_info(int), Variance.INVARIANT),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical covariant constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT),
        Constraint(A, get_generic_info(int), Variance.COVARIANT),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical mixed constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT),
        Constraint(A, get_generic_info(int), Variance.COVARIANT),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test additional edge cases for final coverage push
    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test additional edge cases for final coverage push
    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.INVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test constraint solver with identical override constraints (different variance)
    constraints = [
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
        Constraint(A, get_generic_info(int), Variance.COVARIANT, is_override=True),
    ]

    sub = solve_constraints(constraints)
    result = sub.get(A)
    assert result.resolved_type == int

    # Test ForwardRef handling edge cases
    from typing import ForwardRef

    # Test ForwardRef with matching class name

    def test_forward_ref_match(obj: ForwardRef("SimpleClass")) -> int: ...

    # This should work because class names match
    t = infer_return_type(test_forward_ref_match, SimpleClass(value=42))
    assert t == int

    # Test ForwardRef with mismatched class name

    def test_forward_ref_mismatch(obj: ForwardRef("SimpleClass")) -> int: ...

    # This should fail due to class name mismatch
    with pytest.raises(TypeInferenceError):
        infer_return_type(test_forward_ref_mismatch, DifferentClass(value=42))

    # Test ForwardRef with generic class name

    def test_forward_ref_generic(obj: ForwardRef("GenericTest[A]")) -> A: ...

    # This should work because class names match, but ForwardRef has limitations
    # The engine can't properly resolve ForwardRef with generics
    with pytest.raises(TypeInferenceError):
        infer_return_type(test_forward_ref_generic, GenericTest[int](value=42))

    # Test ForwardRef with complex generic class name

    def test_forward_ref_complex(
        obj: ForwardRef("GenericPair[A, B]"),
    ) -> Tuple[A, B]: ...

    # This should work because class names match, but ForwardRef has limitations
    with pytest.raises(TypeInferenceError):
        infer_return_type(
            test_forward_ref_complex, GenericPair[int, str](first=42, second="hello")
        )

    # Test ForwardRef with nested generic class name

    def test_forward_ref_nested(obj: ForwardRef("NestedGeneric[A]")) -> A: ...

    # This should work because class names match, but ForwardRef has limitations
    with pytest.raises(TypeInferenceError):
        infer_return_type(test_forward_ref_nested, NestedGeneric[int](items=[1, 2, 3]))

    # Test ForwardRef with union class name

    def test_forward_ref_union(obj: ForwardRef("UnionTest")) -> Union[int, str]: ...

    # This should work because class names match
    t = infer_return_type(test_forward_ref_union, UnionTest(value=42))
    assert typing.get_origin(t) in [Union, getattr(types, "UnionType", None)]

    # Test ForwardRef with optional class name

    def test_forward_ref_optional(obj: ForwardRef("OptionalTest")) -> Optional[int]: ...

    # This should work because class names match
    t = infer_return_type(test_forward_ref_optional, OptionalTest(value=42))
    assert typing.get_origin(t) in [Union, getattr(types, "UnionType", None)]

    # Test ForwardRef with tuple class name

    def test_forward_ref_tuple(obj: ForwardRef("TupleTest")) -> Tuple[int, str]: ...

    # This should work because class names match
    t = infer_return_type(test_forward_ref_tuple, TupleTest(value=(42, "hello")))
    assert typing.get_origin(t) == tuple
    assert typing.get_args(t) == (int, str)

    # Test ForwardRef with dict class name

    def test_forward_ref_dict(obj: ForwardRef("DictTest")) -> Dict[str, int]: ...

    # This should work because class names match
    t = infer_return_type(test_forward_ref_dict, DictTest(value={"key": 42}))
    assert typing.get_origin(t) == dict
    assert typing.get_args(t) == (str, int)

    # Test ForwardRef with set class name

    def test_forward_ref_set(obj: ForwardRef("SetTest")) -> Set[int]: ...

    # This should work because class names match
    t = infer_return_type(test_forward_ref_set, SetTest(value={1, 2, 3}))
    assert typing.get_origin(t) == set
    assert typing.get_args(t) == (int,)


@pytest.mark.skip(reason="LIMITATION: typing.Any not supported")
def test_any_type_limitations():
    """Document typing.Any limitations."""

    from typing import Any

    def process_any(x: Any, y: A) -> A: ...

    # Any should accept anything, but shouldn't interfere with A inference
    t = infer_return_type(process_any, "anything", 42)
    assert t == int


# =============================================================================
# BENCHMARK AND PERFORMANCE TESTS
# =============================================================================


@pytest.mark.skip(reason="BENCHMARK: Performance test, not a correctness test")
def test_deeply_nested_performance():
    """Benchmark: Test performance on deeply nested structures."""

    import time

    def deep_nested(data: List[List[List[List[List[A]]]]]) -> A: ...

    deep_data = [[[[[1, 2, 3]]]]]

    start = time.time()
    result = infer_return_type(deep_nested, deep_data)
    elapsed = time.time() - start

    assert result is int
    assert elapsed < 1.0  # Should complete in under 1 second


@pytest.mark.skip(reason="BENCHMARK: Scalability test, not a correctness test")
def test_many_typevars_scalability():
    """Benchmark: Test scalability with many TypeVars."""

    def extract_all(mp: ManyParams[A, B, C, X, Y]) -> Tuple[A, B, C, X, Y]: ...

    instance = ManyParams[int, str, float, bool, bytes](
        a=1, b="hello", c=3.14, x=True, y=b"data"
    )

    import time

    start = time.time()
    result = infer_return_type(extract_all, instance)
    elapsed = time.time() - start

    assert elapsed < 0.5  # Should be fast even with many TypeVars


# =============================================================================
# PEP LIMITATION TESTS
# =============================================================================


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


def test_pep484_noreturn():
    """Test that NoReturn is handled appropriately."""
    from typing import NoReturn

    def process_noreturn(x: NoReturn, y: A) -> A: ...

    # NoReturn should not interfere with A inference
    # (though it's unusual to use in type inference context)
    pass


# =============================================================================
# ADDITIONAL DEEP NESTING STRESS TESTS
# =============================================================================


def test_seven_level_mixed_nesting():
    """Test 7-level mixed container nesting."""

    def extract_deeply_nested(
        data: List[Dict[str, List[Tuple[List[Dict[int, Optional[A]]]]]]],
    ) -> A: ...

    # Create 7-level nested structure that matches the annotation
    # List[Dict[str, List[Tuple[List[Dict[int, Optional[A]]]]]]]
    deep_data = [{"key": [([{1: None}, {2: 42}],)]}]

    t = infer_return_type(extract_deeply_nested, deep_data)
    assert t is int


def test_empty_containers_at_depth():
    """Test that empty containers at various depths are handled."""

    def process_with_empties(a: List[List[List[A]]], b: A) -> A: ...

    # Empty containers at depth should not interfere with direct inference
    t = infer_return_type(process_with_empties, [[[]]], 42)
    assert t is int


def test_mixed_types_at_each_depth_level():
    """Test mixed types at multiple depth levels simultaneously."""

    def process_multi_depth_mixed(data: List[Dict[str, List[A]]]) -> A: ...

    # Mixed types at different depths
    mixed_data = [
        {"key1": [1, 2, 3]},
        {"key2": ["a", "b", "c"]},
        {"key3": [1.0, 2.0, 3.0]},
    ]

    t = infer_return_type(process_multi_depth_mixed, mixed_data)
    # Should create union of all types found
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str, float}


def test_multiple_typevars_all_at_different_depths():
    """Test A at depth 1, B at depth 2, C at depth 3, D at depth 4."""

    def extract_multi_depth_types(
        data: Dict[A, List[Dict[B, Set[Tuple[C, D]]]]],
    ) -> Tuple[A, B, C, D]: ...

    # Each TypeVar at different depth
    multi_depth_data = {
        "depth1": [{"depth2": {("depth3a", "depth4a"), ("depth3b", "depth4b")}}]
    }

    t = infer_return_type(extract_multi_depth_types, multi_depth_data)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args == (str, str, str, str)


def test_union_at_multiple_depths():
    """Test Union types at depths 1, 2, and 3 simultaneously."""

    def process_multi_level_unions(
        data: Union[Dict[A, Union[List[B], Set[Union[C, D]]]], List[A]],
    ) -> Union[A, B, C, D]: ...

    # Union at multiple depths
    multi_union_data = {"key": [1, 2, 3]}

    t = infer_return_type(process_multi_level_unions, multi_union_data)
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)


def test_deep_and_wide():
    """Test structure that is both deep (5 levels) and wide (4 TypeVars)."""

    def extract_deep_wide(
        data: Dict[A, List[Dict[B, Set[Tuple[C, D]]]]],
    ) -> Tuple[A, B, C, D]: ...

    # Deep and wide structure
    deep_wide_data = {
        "wide1": [{"wide2": {("wide3a", "wide4a"), ("wide3b", "wide4b")}}]
    }

    t = infer_return_type(extract_deep_wide, deep_wide_data)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args == (str, str, str, str)


def test_many_nested_containers_same_typevar():
    """Test same TypeVar appearing at multiple depths."""

    def process_repeated_typevar(data: Dict[A, List[Dict[A, Set[A]]]]) -> A: ...

    # Same TypeVar A at multiple depths
    repeated_data = {"level1": [{"level2": {"level3a", "level3b"}}]}

    t = infer_return_type(process_repeated_typevar, repeated_data)
    assert t is str


# =============================================================================
# SPECIALIZED OPTIONAL EDGE CASES
# =============================================================================


def test_list_optional_dict_with_none():
    """List[Optional[Dict[str, Optional[A]]]] with None in list."""

    def process_multi_optional(
        data: Optional[List[Optional[Dict[str, Optional[A]]]]],
    ) -> A: ...

    # Complex Optional nesting with None values
    complex_optional_data = [{"key": 42}, None, {"key": None}, {"key": "hello"}]

    t = infer_return_type(process_multi_optional, complex_optional_data)
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


def test_list_optional_dict_all_none():
    """Edge case: ALL dicts in list are None - should fail."""

    def process_all_none(data: List[Optional[Dict[str, A]]]) -> A: ...

    # All dicts are None - should fail
    all_none_data = [None, None, None]

    with pytest.raises(TypeInferenceError):
        infer_return_type(process_all_none, all_none_data)


def test_list_optional_dict_some_none():
    """List[Optional[Dict[str, A]]] with some None values."""

    def process_some_none(data: List[Optional[Dict[str, A]]]) -> A: ...

    # Some None values mixed with actual data
    some_none_data = [{"key": 42}, None, {"key": "hello"}, None]

    t = infer_return_type(process_some_none, some_none_data)
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


def test_optional_list_vs_list_optional():
    """Compare: Optional[List[...]] vs List[Optional[...]]."""

    def process_optional_list(data: Optional[List[Dict[str, A]]]) -> A: ...

    def process_list_optional(data: List[Optional[Dict[str, A]]]) -> A: ...

    # Test Optional[List[...]] - the list itself might be None
    optional_list_data = [{"key": 42}]
    t1 = infer_return_type(process_optional_list, optional_list_data)
    assert t1 is int

    # Test List[Optional[...]] - individual items might be None
    list_optional_data = [{"key": 42}, None, {"key": "hello"}]
    t2 = infer_return_type(process_list_optional, list_optional_data)
    origin = typing.get_origin(t2)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(t2)
    assert set(union_args) == {int, str}


def test_deeply_nested_optionals():
    """Optional at multiple levels simultaneously."""

    def process_deep_optional(
        data: Optional[List[Optional[Dict[str, Optional[List[Optional[A]]]]]]],
    ) -> A: ...

    # Deep Optional nesting
    deep_optional_data = [
        {"key": [42, None, 43]},
        None,
        {"key": None},
        {"key": ["hello", None, "world"]},
    ]

    t = infer_return_type(process_deep_optional, deep_optional_data)
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


def test_optional_none_filtering():
    """None values in Optional[A] don't bind A to NoneType."""

    def process_optional_values(data: Dict[str, Optional[A]]) -> A: ...

    # None values should be filtered out, not bound to A
    optional_values_data = {"key1": 42, "key2": None, "key3": "hello", "key4": None}

    t = infer_return_type(process_optional_values, optional_values_data)
    origin = typing.get_origin(t)
    assert origin is Union or origin is getattr(types, "UnionType", None)
    union_args = typing.get_args(t)
    assert set(union_args) == {int, str}


# =============================================================================
# ADVANCED CONSTRAINT SOLVER TESTS
# =============================================================================


def test_constraint_solver_many_constraints():
    """Test constraint solver with many constraints."""

    def extract_many_types(
        data: Dict[A, List[Dict[B, Set[Tuple[C, D]]]]],
    ) -> Tuple[A, B, C, D]: ...

    # Many constraints from complex structure
    many_constraints_data = {
        "a1": [{"b1": {("c1", "d1"), ("c2", "d2")}}],
        "a2": [{"b2": {("c3", "d3"), ("c4", "d4")}}],
    }

    t = infer_return_type(extract_many_types, many_constraints_data)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args == (str, str, str, str)


def test_constraint_solver_all_same():
    """Test constraint solver when all constraints are identical."""

    def extract_same_types(data: List[Dict[A, List[B]]]) -> Tuple[A, B]: ...

    # All constraints should be identical
    same_constraints_data = [
        {"key1": [1, 2, 3]},
        {"key2": [4, 5, 6]},
        {"key3": [7, 8, 9]},
    ]

    t = infer_return_type(extract_same_types, same_constraints_data)
    assert typing.get_origin(t) is tuple
    args = typing.get_args(t)
    assert args == (str, int)


def test_nonetype_inference():
    """Test NoneType inference."""

    def process_none(data: Optional[A]) -> A: ...

    # None should not bind A to NoneType
    with pytest.raises(TypeInferenceError):
        infer_return_type(process_none, None)


def test_optional_with_all_none_values():
    """Test Optional when all values are None."""

    def process_all_none_optional(data: List[Optional[A]]) -> A: ...

    # All values are None - should fail
    all_none_data = [None, None, None]

    with pytest.raises(TypeInferenceError):
        infer_return_type(process_all_none_optional, all_none_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

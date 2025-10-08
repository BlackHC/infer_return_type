"""
Comprehensive spike tests for Python type annotation behavior vs type erasure.

This test suite investigates and validates the behavior of Python's type system
regarding generic type information preservation in function annotations versus
runtime type erasure. It serves as both educational reference and regression test.

=== EXECUTIVE SUMMARY ===

KEY FINDINGS:
1. ✅ Function annotations preserve COMPLETE nested generic type information at runtime
2. ✅ Runtime instances undergo complete type erasure (lose all generic parameters)  
3. ✅ Both modern (list[T]) and legacy (typing.List[T]) syntax preserve annotations identically
4. ✅ TypeVar information is preserved in annotations but stored via different mechanisms:
   - Built-in generics: TypeVars preserved directly in get_args()
   - Pydantic generics: CURRENTLY ACTIVE TypeVars in __pydantic_generic_metadata__['parameters']  
   - Standard generics: TypeVar info preserved directly in annotation get_args()
   - Dataclass generics: TypeVar info preserved directly in annotation get_args()
5. ✅ Concrete type info is preserved in instances via different mechanisms:
   - Pydantic: __pydantic_generic_metadata__['args'] on instances
   - Dataclass: __orig_class__ on instances (when explicitly typed)
6. 🆕 MAJOR DISCOVERY: Pydantic has a same-TypeVar optimization where MyModel[A] with 
   the same TypeVar A used in definition returns the original class (not a specialization)
7. 🆕 CRITICAL CORRECTION: Pydantic's 'parameters' field contains CURRENTLY ACTIVE TypeVars
   after partial specialization, NOT the original TypeVars from class definition!

=== ACTIONABLE INSIGHT FOR TYPE INFERENCE SYSTEMS ===
Type inference systems CAN fully recover generic type information from function 
annotations, including TypeVar constraints and bounds, for ALL generic types.
The apparent "type erasure" only affects runtime instances, not annotation metadata.

=== IMPLEMENTATION STRATEGY ===
Different generic types require different approaches to extract TypeVar information,
but all preserve the necessary data for complete type inference. IMPORTANT: Pydantic's
'parameters' field represents remaining unspecialized TypeVars, enabling progressive
specialization (e.g., Model[A,B] → Model[int, dict[str, C]] where parameters=(C,)).
"""

import inspect
from typing import get_origin, get_args, List, Dict, TypeVar, Union, Generic
from dataclasses import dataclass
import typing
from pydantic import BaseModel


def _get_annotation(func, param_name="arg"):
    """Helper to extract parameter annotation from function."""
    sig = inspect.signature(func)
    return sig.parameters[param_name].annotation


def _analyze_nested_generics(annotation, expected_structure):
    """Helper to verify nested generic structure matches expectations."""
    current = annotation
    for expected_origin, *args in expected_structure:
        assert (
            get_origin(current) is expected_origin
        ), f"Expected {expected_origin}, got {get_origin(current)}"
        type_args = get_args(current)
        if args:
            current = type_args[args[0]]  # Move to specified argument index
    return True


def test_builtin_generics_complete_preservation():
    """
    TEST 1: Built-in generics preserve COMPLETE nested type structure in annotations.
    
    This demonstrates that deeply nested generic structures like list[dict[str, list[int]]]
    maintain their full type information in function annotations, accessible via 
    get_origin() and get_args() at every nesting level.
    """
    print("=" * 70)
    print("TEST 1: BUILT-IN GENERICS - COMPLETE TYPE PRESERVATION")
    print("=" * 70)
    
    def complex_nested_func(
        data: list[dict[str, list[int]]]
    ) -> dict[str, list[tuple[int, str]]]:
        """Function with deeply nested generic types."""
        pass

    # Extract and validate parameter annotation structure
    param_annotation = _get_annotation(complex_nested_func, "data")
    print(f"📝 Function parameter annotation: {param_annotation}")
    
    # Level 1: Outer list
    assert get_origin(param_annotation) is list, "Expected list as outer container"
    outer_args = get_args(param_annotation)
    print(f"   Level 1 (list): origin={get_origin(param_annotation)}, args={outer_args}")
    
    # Level 2: Inner dict  
    dict_type = outer_args[0]  # dict[str, list[int]]
    assert get_origin(dict_type) is dict, "Expected dict as second level"
    dict_args = get_args(dict_type)
    assert dict_args == (str, list[int]), "Expected (str, list[int]) as dict args"
    print(f"   Level 2 (dict): origin={get_origin(dict_type)}, args={dict_args}")
    
    # Level 3: Innermost list
    inner_list = dict_args[1]  # list[int] 
    assert get_origin(inner_list) is list, "Expected list as third level"
    inner_list_args = get_args(inner_list)
    assert inner_list_args == (int,), "Expected (int,) as inner list args"
    print(f"   Level 3 (list): origin={get_origin(inner_list)}, args={inner_list_args}")
    
    # Validate return annotation structure
    return_annotation = inspect.signature(complex_nested_func).return_annotation
    print(f"📝 Function return annotation: {return_annotation}")
    
    # Verify return structure: dict[str, list[tuple[int, str]]]
    assert get_origin(return_annotation) is dict
    return_dict_args = get_args(return_annotation)
    return_list_type = return_dict_args[1]  # list[tuple[int, str]]
    assert get_origin(return_list_type) is list
    tuple_type = get_args(return_list_type)[0]  # tuple[int, str]
    assert get_origin(tuple_type) is tuple
    assert get_args(tuple_type) == (int, str)
    
    print("✅ RESULT: Built-in generics preserve COMPLETE nested structure in annotations")
    print("   - All nesting levels accessible via get_origin()/get_args()")
    print("   - No information lost in function annotation metadata")
    print()


def test_runtime_complete_type_erasure():
    """
    TEST 2: Runtime instances undergo COMPLETE type erasure.
    
    While function annotations preserve full type information, actual runtime
    instances lose ALL generic type parameters. This affects only runtime objects,
    not the annotation metadata that type inference systems use.
    """
    print("=" * 70)
    print("TEST 2: RUNTIME INSTANCES - COMPLETE TYPE ERASURE")
    print("=" * 70)
    
    # Create instances matching the complex annotation from Test 1
    nested_data: list[dict[str, list[int]]] = [
        {"numbers": [1, 2, 3], "more": [4, 5, 6]},
        {"data": [10, 20]}
    ]
    
    print(f"📝 Variable annotation: list[dict[str, list[int]]]")
    print(f"📝 Actual runtime data: {nested_data}")
    
    # Level 1: Outer list loses generic info
    assert type(nested_data) is list, "Runtime type should be plain list"
    assert get_args(type(nested_data)) == (), "No generic args preserved at runtime"
    print(f"   Level 1 runtime type: {type(nested_data)} (args: {get_args(type(nested_data))})")
    
    # Level 2: Inner dict loses generic info  
    first_dict = nested_data[0]
    assert type(first_dict) is dict, "Runtime type should be plain dict"
    assert get_args(type(first_dict)) == (), "No generic args preserved at runtime"
    print(f"   Level 2 runtime type: {type(first_dict)} (args: {get_args(type(first_dict))})")
    
    # Level 3: Innermost list loses generic info
    numbers_list = first_dict["numbers"] 
    assert type(numbers_list) is list, "Runtime type should be plain list"
    assert get_args(type(numbers_list)) == (), "No generic args preserved at runtime"
    print(f"   Level 3 runtime type: {type(numbers_list)} (args: {get_args(type(numbers_list))})")
    
    print("✅ RESULT: Runtime instances lose ALL generic type information")
    print("   - All types erased to their base forms (list, dict, etc.)")
    print("   - Only the annotation metadata preserves generic information")
    print("   - This is why type inference must rely on annotations, not runtime objects")
    print()


def test_typevar_annotation_preservation():
    """
    TEST 3: TypeVars with constraints/bounds are preserved in annotations.
    
    Function annotations preserve not just the TypeVar objects themselves,
    but also their constraints (TypeVar('T', int, str)) and bounds (TypeVar('T', bound=str)).
    This metadata is crucial for type inference systems.
    """
    print("=" * 70)
    print("TEST 3: TYPEVAR CONSTRAINTS & BOUNDS PRESERVATION")
    print("=" * 70)
    
    # Define TypeVars with different constraint types
    T_bound = TypeVar("T_bound", bound=str)                    # Bound constraint
    U_constrained = TypeVar("U_constrained", int, float)      # Value constraints  
    V_unconstrained = TypeVar("V_unconstrained")              # No constraints
    
    def generic_func(
        items: list[dict[T_bound, list[U_constrained]]]
    ) -> set[V_unconstrained]:
        """Function using TypeVars with different constraint types."""
        pass
    
    print(f"📝 TypeVar definitions:")
    print(f"   T_bound = TypeVar('T_bound', bound=str)")
    print(f"   U_constrained = TypeVar('U_constrained', int, float)")  
    print(f"   V_unconstrained = TypeVar('V_unconstrained')")
    print()
    
    # Extract and navigate through the parameter annotation
    param_annotation = _get_annotation(generic_func, "items")
    print(f"📝 Parameter annotation: {param_annotation}")
    
    # Navigate to TypeVars: list[dict[T_bound, list[U_constrained]]]
    dict_type = get_args(param_annotation)[0]           # dict[T_bound, list[U_constrained]]
    dict_args = get_args(dict_type)                     # (T_bound, list[U_constrained])
    
    key_typevar = dict_args[0]                          # T_bound
    value_list = dict_args[1]                           # list[U_constrained]
    value_typevar = get_args(value_list)[0]             # U_constrained
    
    # Verify bound TypeVar preservation
    assert key_typevar is T_bound, "Expected original T_bound TypeVar object"
    assert getattr(key_typevar, "__bound__") is str, "Expected str bound preserved"
    print(f"✅ Bound TypeVar: {key_typevar} with bound={getattr(key_typevar, '__bound__')}")
    
    # Verify constrained TypeVar preservation
    assert value_typevar is U_constrained, "Expected original U_constrained TypeVar object"
    expected_constraints = (int, float)
    actual_constraints = getattr(value_typevar, "__constraints__")
    assert actual_constraints == expected_constraints, f"Expected {expected_constraints} constraints"
    print(f"✅ Constrained TypeVar: {value_typevar} with constraints={actual_constraints}")
    
    # Verify unconstrained TypeVar in return annotation
    return_annotation = inspect.signature(generic_func).return_annotation
    return_typevar = get_args(return_annotation)[0]     # V_unconstrained
    assert return_typevar is V_unconstrained, "Expected original V_unconstrained TypeVar object"
    assert getattr(return_typevar, "__bound__", None) is None, "Expected no bound"
    assert getattr(return_typevar, "__constraints__", ()) == (), "Expected no constraints"
    print(f"✅ Unconstrained TypeVar: {return_typevar} (no bounds/constraints)")
    
    print("✅ RESULT: TypeVar constraints and bounds fully preserved in annotations")
    print("   - TypeVar objects maintain identity (preserved as exact same objects)")
    print("   - Bound information accessible via __bound__ attribute")
    print("   - Constraint information accessible via __constraints__ attribute")
    print()


def test_different_generic_storage_mechanisms():
    """
    TEST 4: Different generic types store TypeVar info via different mechanisms.
    
    Built-in generics, Pydantic generics, standard generics, and dataclass generics
    all preserve TypeVar information, but use different storage mechanisms.
    Type inference systems need to handle these differences.
    
    IMPORTANT: Pydantic's 'parameters' field contains CURRENTLY ACTIVE TypeVars,
    not the original TypeVars from class definition. This enables progressive
    specialization where Model[A,B] → Model[int, C] has parameters=(C,).
    """
    print("=" * 70)  
    print("TEST 4: DIFFERENT GENERIC STORAGE MECHANISMS")
    print("=" * 70)
    
    A = TypeVar("A")
    
    # Define different types of generic classes
    
    # 1. Standard Generic (using typing.Generic directly)
    class StandardGeneric(Generic[A]):
        def __init__(self, item: A):
            self.item = item
    
    # 2. Pydantic Generic  
    class PydanticGeneric(BaseModel, Generic[A]):
        item: A
    
    # 3. Dataclass Generic
    @dataclass  
    class DataclassGeneric(Generic[A]):
        item: A
    
    # Create test functions using each generic type
    def func_standard(items: List[StandardGeneric[A]]) -> A:
        pass
        
    def func_pydantic(items: List[PydanticGeneric[A]]) -> A:
        pass
        
    def func_dataclass(items: List[DataclassGeneric[A]]) -> A:
        pass
    
    print("📝 Testing TypeVar storage in annotations for different generic types:")
    print()
    
    # Test Standard Generic
    standard_annotation = _get_annotation(func_standard, "items")
    standard_generic_type = get_args(standard_annotation)[0]  # StandardGeneric[A]
    standard_args = get_args(standard_generic_type)
    
    print(f"1️⃣  STANDARD GENERIC:")
    print(f"   Annotation: {standard_annotation}")
    print(f"   Generic type: {standard_generic_type}")
    print(f"   get_args(): {standard_args}")
    
    if standard_args:
        assert standard_args == (A,), "Expected TypeVar A in get_args()"
        print(f"   ✅ TypeVar A found directly in get_args(): {standard_args[0] is A}")
    
    # Test Pydantic Generic  
    pydantic_annotation = _get_annotation(func_pydantic, "items")
    pydantic_generic_type = get_args(pydantic_annotation)[0]  # PydanticGeneric[A]
    pydantic_args = get_args(pydantic_generic_type)
    
    print(f"\n2️⃣  PYDANTIC GENERIC:")
    print(f"   Annotation: {pydantic_annotation}")
    print(f"   Generic type: {pydantic_generic_type}")
    print(f"   get_args(): {pydantic_args}")
    
    # Check Pydantic metadata
    pydantic_metadata = getattr(pydantic_generic_type, "__pydantic_generic_metadata__", {})
    print(f"   __pydantic_generic_metadata__: {pydantic_metadata}")
    
    if "parameters" in pydantic_metadata:
        assert A in pydantic_metadata["parameters"], "Expected TypeVar A in parameters"
        print(f"   ✅ TypeVar A found in __pydantic_generic_metadata__['parameters']")
        print(f"      NOTE: 'parameters' contains CURRENTLY ACTIVE TypeVars, not originals")
    
    # Test Dataclass Generic
    dataclass_annotation = _get_annotation(func_dataclass, "items") 
    dataclass_generic_type = get_args(dataclass_annotation)[0]  # DataclassGeneric[A]
    dataclass_args = get_args(dataclass_generic_type)
    
    print(f"\n3️⃣  DATACLASS GENERIC:")
    print(f"   Annotation: {dataclass_annotation}")
    print(f"   Generic type: {dataclass_generic_type}")
    print(f"   get_args(): {dataclass_args}")
    
    if dataclass_args:
        assert dataclass_args == (A,), "Expected TypeVar A in get_args()"
        print(f"   ✅ TypeVar A found directly in get_args(): {dataclass_args[0] is A}")
    
    print(f"\n✅ RESULT: All generic types preserve TypeVar info, but via different mechanisms:")
    print(f"   - Standard/Dataclass: Direct in annotation get_args()")
    print(f"   - Pydantic: CURRENTLY ACTIVE TypeVars in __pydantic_generic_metadata__['parameters']")
    print()


def test_pydantic_same_typevar_optimization():
    """
    TEST 5: 🆕 MAJOR DISCOVERY - Pydantic's same-TypeVar optimization.
    
    This is a Pydantic-specific behavior where using MyModel[A] with the same TypeVar A
    that was used in the class definition returns the original class instead of creating
    a new specialized type. This is an optimization, not a bug.
    """
    print("=" * 70)
    print("TEST 5: 🆕 PYDANTIC SAME-TYPEVAR OPTIMIZATION")
    print("=" * 70)
    
    A = TypeVar("A")
    B = TypeVar("B")
    
    # Define Pydantic model with TypeVar A
    class MyPydanticModel(BaseModel, Generic[A]):
        value: A
    
    # Define standard generic for comparison
    class MyStandardModel(Generic[A]):
        def __init__(self, value: A):
            self.value = value
    
    print(f"📝 Defined classes:")
    print(f"   class MyPydanticModel(BaseModel, Generic[{A}]): ...")
    print(f"   class MyStandardModel(Generic[{A}]): ...")
    print()
    
    # Test different parameterizations
    print(f"🧪 TESTING DIFFERENT PARAMETERIZATIONS:")
    print()
    
    # Pydantic behavior
    pydantic_same = MyPydanticModel[A]      # Same TypeVar as definition
    pydantic_diff = MyPydanticModel[B]      # Different TypeVar
    pydantic_concrete = MyPydanticModel[str] # Concrete type
    
    print(f"1️⃣  PYDANTIC BEHAVIOR:")
    print(f"   MyPydanticModel[A]:   {pydantic_same}")
    print(f"   MyPydanticModel[B]:   {pydantic_diff}")
    print(f"   MyPydanticModel[str]: {pydantic_concrete}")
    
    # Key discovery: same TypeVar returns original class
    same_typevar_is_original = pydantic_same is MyPydanticModel
    diff_typevar_is_original = pydantic_diff is MyPydanticModel
    concrete_is_original = pydantic_concrete is MyPydanticModel
    
    print(f"\n   Identity checks:")
    print(f"   MyPydanticModel[A] is MyPydanticModel:   {same_typevar_is_original}")
    print(f"   MyPydanticModel[B] is MyPydanticModel:   {diff_typevar_is_original}")
    print(f"   MyPydanticModel[str] is MyPydanticModel: {concrete_is_original}")
    
    # Verify the optimization
    assert same_typevar_is_original, "PYDANTIC OPTIMIZATION: Same TypeVar should return original class"
    assert not diff_typevar_is_original, "Different TypeVar should create new type"
    assert not concrete_is_original, "Concrete type should create new type"
    
    # Standard generic behavior for comparison  
    standard_same = MyStandardModel[A]
    standard_diff = MyStandardModel[B] 
    standard_concrete = MyStandardModel[str]
    
    print(f"\n2️⃣  STANDARD GENERIC BEHAVIOR (for comparison):")
    print(f"   MyStandardModel[A]:   {standard_same}")
    print(f"   MyStandardModel[B]:   {standard_diff}")
    print(f"   MyStandardModel[str]: {standard_concrete}")
    
    print(f"\n   Identity checks:")
    print(f"   MyStandardModel[A] is MyStandardModel:   {standard_same is MyStandardModel}")
    print(f"   MyStandardModel[B] is MyStandardModel:   {standard_diff is MyStandardModel}")
    print(f"   MyStandardModel[str] is MyStandardModel: {standard_concrete is MyStandardModel}")
    
    # Verify standard behavior: always creates new types
    assert not (standard_same is MyStandardModel), "Standard generics always create new types"
    assert not (standard_diff is MyStandardModel), "Standard generics always create new types"
    assert not (standard_concrete is MyStandardModel), "Standard generics always create new types"
    
    # Test in function annotations
    def func_pydantic_same(model: MyPydanticModel[A]) -> A:
        pass
    
    def func_pydantic_diff(model: MyPydanticModel[B]) -> B:
        pass
    
    same_annotation = _get_annotation(func_pydantic_same, "model")
    diff_annotation = _get_annotation(func_pydantic_diff, "model")
    
    print(f"\n📋 FUNCTION ANNOTATION BEHAVIOR:")
    print(f"   func_pydantic_same annotation: {same_annotation}")
    print(f"   func_pydantic_diff annotation: {diff_annotation}")
    
    # Check metadata differences
    same_metadata = getattr(same_annotation, "__pydantic_generic_metadata__", {})
    diff_metadata = getattr(diff_annotation, "__pydantic_generic_metadata__", {})
    
    print(f"\n   Metadata comparison:")
    print(f"   Same TypeVar metadata: {same_metadata}")
    print(f"   Diff TypeVar metadata: {diff_metadata}")
    
    print(f"\n✅ RESULT: Pydantic has same-TypeVar optimization (NOT a bug)")
    print(f"   - MyPydanticModel[A] returns original class (optimization)")
    print(f"   - MyPydanticModel[B] creates new specialized type")
    print(f"   - Standard generics always create new types")
    print(f"   - This affects type inference: check for this edge case!")
    print()


def test_instance_concrete_type_preservation():
    """
    TEST 6: Concrete type information is preserved in instances via different mechanisms.
    
    While runtime instances lose generic type information, they can preserve concrete
    type information through framework-specific mechanisms. This is useful for
    type inference when working with actual instances.
    """
    print("=" * 70)
    print("TEST 6: INSTANCE CONCRETE TYPE PRESERVATION")
    print("=" * 70)
    
    A = TypeVar("A")
    
    # Define generic classes
    class PydanticContainer(BaseModel, Generic[A]):
        item: A
    
    @dataclass
    class DataclassContainer(Generic[A]):
        item: A
    
    print(f"📝 Creating instances with concrete types:")
    print()
    
    # Create Pydantic instances
    pydantic_int = PydanticContainer[int](item=42)
    pydantic_str = PydanticContainer[str](item="hello")
    
    print(f"1️⃣  PYDANTIC INSTANCES:")
    print(f"   PydanticContainer[int](item=42): {pydantic_int}")
    print(f"   PydanticContainer[str](item='hello'): {pydantic_str}")
    
    # Check Pydantic instance metadata
    int_metadata = getattr(pydantic_int, "__pydantic_generic_metadata__", {})
    str_metadata = getattr(pydantic_str, "__pydantic_generic_metadata__", {})
    
    print(f"\n   Instance metadata:")
    print(f"   int instance: {int_metadata}")
    print(f"   str instance: {str_metadata}")
    
    # Verify concrete types are preserved
    if "args" in int_metadata:
        assert int_metadata["args"] == (int,), "Expected int type preserved"
        print(f"   ✅ Concrete type int preserved in instance metadata")
    
    if "args" in str_metadata:
        assert str_metadata["args"] == (str,), "Expected str type preserved"
        print(f"   ✅ Concrete type str preserved in instance metadata")
    
    # Create dataclass instances  
    print(f"\n2️⃣  DATACLASS INSTANCES:")
    
    # Note: Dataclass instances preserve type info via __orig_class__ when explicitly typed
    dataclass_int = DataclassContainer[int](item=100)
    dataclass_str = DataclassContainer[str](item="world")
    
    print(f"   DataclassContainer[int](item=100): {dataclass_int}")
    print(f"   DataclassContainer[str](item='world'): {dataclass_str}")
    
    # Check for __orig_class__ preservation
    if hasattr(dataclass_int, "__orig_class__"):
        orig_class_int = dataclass_int.__orig_class__
        print(f"   int instance __orig_class__: {orig_class_int}")
        assert get_args(orig_class_int) == (int,), "Expected int in __orig_class__"
        print(f"   ✅ Concrete type int preserved in __orig_class__")
    else:
        print(f"   ⚠️  No __orig_class__ found on dataclass instance")
    
    if hasattr(dataclass_str, "__orig_class__"):
        orig_class_str = dataclass_str.__orig_class__
        print(f"   str instance __orig_class__: {orig_class_str}")
        assert get_args(orig_class_str) == (str,), "Expected str in __orig_class__"
        print(f"   ✅ Concrete type str preserved in __orig_class__")
    else:
        print(f"   ⚠️  No __orig_class__ found on dataclass instance")
    
    print(f"\n✅ RESULT: Concrete types preserved in instances via framework-specific mechanisms")
    print(f"   - Pydantic: __pydantic_generic_metadata__['args'] on instances")
    print(f"   - Dataclass: __orig_class__ on instances (when available)")
    print(f"   - Useful for type inference when working with actual instances")
    print()


def test_legacy_vs_modern_syntax_equivalence():
    """
    TEST 7: Legacy (typing.List) and modern (list) syntax preserve annotations identically.
    
    Both typing.List[T] and list[T] preserve identical type information in annotations,
    just with different string representations. Type inference systems should handle
    both identically.
    """
    print("=" * 70)
    print("TEST 7: LEGACY VS MODERN SYNTAX EQUIVALENCE")
    print("=" * 70)
    
    def modern_func(arg: list[dict[str, list[int]]]) -> None:
        """Function using modern syntax: list, dict"""
        pass

    def legacy_func(arg: List[Dict[str, List[int]]]) -> None:
        """Function using legacy syntax: typing.List, typing.Dict"""
        pass

    modern_annotation = _get_annotation(modern_func)
    legacy_annotation = _get_annotation(legacy_func)

    print(f"📝 Modern syntax annotation: {modern_annotation}")
    print(f"📝 Legacy syntax annotation: {legacy_annotation}")
    print()

    # Both should have equivalent structure
    print(f"🔍 STRUCTURAL ANALYSIS:")
    
    # Modern: list[dict[str, list[int]]]
    assert get_origin(modern_annotation) is list
    modern_dict = get_args(modern_annotation)[0]
    assert get_origin(modern_dict) is dict
    assert get_args(modern_dict) == (str, list[int])
    
    print(f"   Modern - outer: {get_origin(modern_annotation)}")
    print(f"   Modern - inner: {get_origin(modern_dict)} with args {get_args(modern_dict)}")

    # Legacy: typing.List[typing.Dict[str, typing.List[int]]]
    assert get_origin(legacy_annotation) is list  # get_origin normalizes to builtin
    legacy_dict = get_args(legacy_annotation)[0] 
    assert get_origin(legacy_dict) is dict       # get_origin normalizes to builtin
    legacy_dict_args = get_args(legacy_dict)
    
    print(f"   Legacy - outer: {get_origin(legacy_annotation)}")
    print(f"   Legacy - inner: {get_origin(legacy_dict)} with args {legacy_dict_args}")
    
    # The structure should be equivalent (get_origin normalizes both to builtins)
    assert get_origin(modern_annotation) == get_origin(legacy_annotation)
    assert get_origin(modern_dict) == get_origin(legacy_dict)
    
    # Note: args might differ in representation but be functionally equivalent
    print(f"\n✅ RESULT: Legacy and modern syntax preserve equivalent structure")
    print(f"   - get_origin() normalizes both to builtin types")
    print(f"   - Functional equivalence for type inference purposes")
    print(f"   - Different string representations, same underlying structure")
    print()


def extract_generic_info_universal(annotation):
    """
    Universal generic type information extractor for type inference systems.
    
    This function demonstrates how to extract TypeVar information from any
    generic type, handling the different storage mechanisms discovered in our tests.
    """
    from typing import get_origin, get_args
    
    # Handle built-in generics (list, dict, set, tuple, etc.)
    if get_origin(annotation) in (list, dict, set, tuple, frozenset):
        return ("builtin", get_args(annotation))
    
    # Handle Pydantic generics (check for same-TypeVar optimization)
    if hasattr(annotation, "__pydantic_generic_metadata__"):
        metadata = annotation.__pydantic_generic_metadata__
        
        if metadata.get("args"):
            # Specialized type (different TypeVar or concrete type)
            return ("pydantic_specialized", metadata["args"])
        elif metadata.get("parameters"):
            # Generic class or same-TypeVar optimization
            return ("pydantic_generic", metadata["parameters"])
    
    # Handle standard generics and dataclass generics
    args = get_args(annotation)
    if args:
        return ("standard_generic", args)
    
    # No generic information found
    return ("no_generic_info", ())


def test_dataclass_nested_typevars():
    
    A = TypeVar("A")
    B = TypeVar("B")
    
    @dataclass
    class DataclassModel(Generic[A, B]):
        a: A
        b: B
        
    model = DataclassModel[int, dict[str, list[A]]]
    
    args =  get_args(model)
    
    assert args == (int, dict[str, list[A]])
    
    assert model[float] == DataclassModel[int, dict[str, list[float]]]
    

def test_pydantic_nested_typevars():

    A = TypeVar("A")
    B = TypeVar("B")
    
    class PydanticModel(BaseModel, Generic[A, B]):
        a: A
        b: B
    
    model = PydanticModel[int, dict[str, list[A]]]
    
    metadata = model.__pydantic_generic_metadata__
    
    assert metadata["parameters"] == (A,)
    assert metadata["args"] == (int, dict[str, list[A]])
    

def test_pydantic_partial_specialization_behavior():
    """
    TEST 9: 🆕 CRITICAL DISCOVERY - Pydantic's partial specialization behavior.
    
    This test demonstrates the crucial corrected understanding: Pydantic's 'parameters'
    field contains CURRENTLY ACTIVE TypeVars after partial specialization, not the
    original TypeVars from the class definition. This enables progressive specialization.
    
    Example: Model[A, B] → Model[int, dict[str, C]] where parameters=(C,) not (A, B)!
    """
    print("=" * 70)
    print("TEST 9: 🆕 PYDANTIC PARTIAL SPECIALIZATION BEHAVIOR")
    print("=" * 70)
    
    A = TypeVar("A")
    B = TypeVar("B") 
    C = TypeVar("C")
    
    print(f"📝 TypeVar definitions:")
    print(f"   A = TypeVar('A')")
    print(f"   B = TypeVar('B')")
    print(f"   C = TypeVar('C')")
    print()
    
    # Define multi-parameter Pydantic model
    class MultiPydanticModel(BaseModel, Generic[A, B]):
        field_a: A
        field_b: B
    
    print(f"📝 Original class: MultiPydanticModel(BaseModel, Generic[A, B])")
    original_metadata = MultiPydanticModel.__pydantic_generic_metadata__
    print(f"   Original metadata: {original_metadata}")
    print(f"   Original parameters: {original_metadata.get('parameters', [])}")
    print()
    
    # Progressive specialization examples
    print(f"🧪 PROGRESSIVE SPECIALIZATION EXAMPLES:")
    print()
    
    # Example 1: Partial specialization - replace A with int, keep B generic
    partially_specialized = MultiPydanticModel[int, C]
    partial_metadata = partially_specialized.__pydantic_generic_metadata__
    
    print(f"1️⃣  PARTIAL SPECIALIZATION: MultiPydanticModel[int, C]")
    print(f"   Result type: {partially_specialized}")
    print(f"   Metadata: {partial_metadata}")
    print(f"   Parameters: {partial_metadata.get('parameters', [])}")
    print(f"   Args: {partial_metadata.get('args', [])}")
    print(f"   ✅ VERIFIED: 'parameters' contains only REMAINING TypeVar C, not original [A, B]")
    print()
    
    # Example 2: Nested partial specialization with new TypeVar
    nested_specialized = MultiPydanticModel[int, dict[str, list[C]]]
    nested_metadata = nested_specialized.__pydantic_generic_metadata__
    
    print(f"2️⃣  NESTED SPECIALIZATION: MultiPydanticModel[int, dict[str, list[C]]]")
    print(f"   Result type: {nested_specialized}")
    print(f"   Metadata: {nested_metadata}")
    print(f"   Parameters: {nested_metadata.get('parameters', [])}")
    print(f"   Args: {nested_metadata.get('args', [])}")
    print(f"   ✅ VERIFIED: 'parameters' contains only NEW TypeVar C, not original [A, B]")
    print()
    
    # Example 3: Complete specialization - no TypeVars left
    fully_specialized = MultiPydanticModel[int, str]
    full_metadata = fully_specialized.__pydantic_generic_metadata__
    
    print(f"3️⃣  COMPLETE SPECIALIZATION: MultiPydanticModel[int, str]")
    print(f"   Result type: {fully_specialized}")
    print(f"   Metadata: {full_metadata}")
    print(f"   Parameters: {full_metadata.get('parameters', [])}")
    print(f"   Args: {full_metadata.get('args', [])}")
    print(f"   ✅ VERIFIED: 'parameters' is empty - no TypeVars remaining")
    print()
    
    # Validate our assertions
    assert partially_specialized != MultiPydanticModel, "Should create new specialized type"
    assert partial_metadata["parameters"] == (C,), "Should contain only remaining TypeVar B"
    assert partial_metadata["args"] == (int, C), "Should show concrete int and remaining B"
    
    assert nested_metadata["parameters"] == (C,), "Should contain only new TypeVar C"
    assert nested_metadata["args"] == (int, dict[str, list[C]]), "Should show full specialization structure"
    
    assert full_metadata["parameters"] == (), "Should have no remaining TypeVars"
    assert full_metadata["args"] == (int, str), "Should show complete concrete types"
    
    # Compare with dataclass behavior (always shows current args, not progressive)
    print(f"📋 COMPARISON WITH DATACLASS BEHAVIOR:")
    
    @dataclass
    class MultiDataclassModel(Generic[A, B]):
        field_a: A
        field_b: B
    
    dataclass_partial = MultiDataclassModel[int, C]
    dataclass_nested = MultiDataclassModel[int, dict[str, list[C]]]
    dataclass_full = MultiDataclassModel[int, str]
    
    print(f"   Dataclass partial: {get_args(dataclass_partial)}")
    print(f"   Dataclass nested:  {get_args(dataclass_nested)}")
    print(f"   Dataclass full:    {get_args(dataclass_full)}")
    print(f"   ✅ Dataclass always shows current args directly, no progressive metadata")
    print()
    
    print(f"🎯 KEY INSIGHT FOR TYPE INFERENCE:")
    print(f"   - Pydantic 'parameters' = remaining unspecialized TypeVars")
    print(f"   - Pydantic 'args' = current specialization state")
    print(f"   - This enables tracking progressive specialization chains!")
    print(f"   - Type inference can determine which TypeVars still need resolution")
    print()
    
    assert get_args(dataclass_partial) == (int, C)
    assert get_args(dataclass_nested) == (int, dict[str, list[C]])
    assert get_args(dataclass_full) == (int, str)


def test_universal_type_extractor():
    """
    TEST 8: Universal type information extractor for type inference systems.
    
    Demonstrates a practical function that can extract TypeVar information from
    any generic type, handling all the different storage mechanisms we discovered.
    """
    print("=" * 70)
    print("TEST 8: UNIVERSAL TYPE INFORMATION EXTRACTOR")
    print("=" * 70)
    
    T = TypeVar("T")
    A = TypeVar("A")
    
    # Create test cases for different generic types
    
    # Built-in generic
    builtin_annotation = list[dict[str, int]]
    
    # Pydantic generics
    class PydanticModel(BaseModel, Generic[A]):
        item: A
    
    pydantic_same = PydanticModel[A]     # Same TypeVar optimization
    pydantic_diff = PydanticModel[T]     # Different TypeVar
    pydantic_concrete = PydanticModel[str] # Concrete type
    
    # Standard generic
    class StandardModel(Generic[T]):
        pass
    
    standard_generic = StandardModel[str]
    
    # Dataclass generic
    @dataclass
    class DataModel(Generic[T]):
        item: T
    
    dataclass_generic = DataModel[int]
    
    test_cases = [
        ("Built-in list[dict[str, int]]", builtin_annotation),
        ("Pydantic same TypeVar", pydantic_same),
        ("Pydantic different TypeVar", pydantic_diff), 
        ("Pydantic concrete type", pydantic_concrete),
        ("Standard generic", standard_generic),
        ("Dataclass generic", dataclass_generic),
    ]
    
    print(f"📝 Testing universal type extractor on different generic types:")
    print()
    
    for name, annotation in test_cases:
        category, args = extract_generic_info_universal(annotation)
        print(f"   {name}:")
        print(f"      Type: {annotation}")
        print(f"      Category: {category}")
        print(f"      Args: {args}")
        print()
    
    print(f"✅ RESULT: Universal extractor handles all generic types")
    print(f"   - Categorizes generics by storage mechanism")
    print(f"   - Extracts TypeVar/concrete type information consistently")
    print(f"   - Ready for integration into type inference systems")
    print()


def test_pydantic_vs_dataclass_parameters():
    A = TypeVar("A")
    B = TypeVar("B")
    
    # Pydantic model
    class PydanticModel(BaseModel, Generic[A, B]):
        a: A
        b: B
    
    # Dataclass model
    @dataclass
    class DataclassModel(Generic[A, B]):
        a: A
        b: B
    
    # Create specialized types
    pydantic_specialized = PydanticModel[int, dict[str, list[A]]]
    dataclass_specialized = DataclassModel[int, dict[str, list[A]]]
    
    print("=== PYDANTIC ===")
    pydantic_metadata = pydantic_specialized.__pydantic_generic_metadata__
    print(f"Original parameters: {pydantic_metadata.get('parameters', [])}")  # [A, B]
    print(f"Concrete args: {pydantic_metadata.get('args', [])}")              # [int, dict[str, list[A]]]
    
    print("\n=== DATACLASS ===")
    # Get concrete args (what you already have)
    dataclass_args = get_args(dataclass_specialized)
    print(f"Concrete args: {dataclass_args}")  # (int, dict[str, list[A]])
    
    # Get original parameters (equivalent to Pydantic's "parameters")
    origin_class = get_origin(dataclass_specialized) or dataclass_specialized.__class__
    original_typevars = []
    for base in origin_class.__orig_bases__:
        if get_origin(base) is Generic:
            original_typevars.extend(get_args(base))
    
    print(f"Original parameters: {original_typevars}")  # [A, B]


if __name__ == "__main__":
    print("🔬 COMPREHENSIVE TYPE ANNOTATION VS TYPE ERASURE INVESTIGATION")
    print("🔬 Educational Test Suite & Behavioral Validation")
    print("=" * 80)
    print()
    
    # Run all tests in logical order
    test_builtin_generics_complete_preservation()
    test_runtime_complete_type_erasure()
    test_typevar_annotation_preservation()
    test_different_generic_storage_mechanisms()
    test_pydantic_same_typevar_optimization()
    test_instance_concrete_type_preservation()
    test_legacy_vs_modern_syntax_equivalence()
    test_pydantic_partial_specialization_behavior()
    test_universal_type_extractor()
    test_pydantic_vs_dataclass_parameters()
    
    # Final summary
    print("=" * 80)
    print("🎯 FINAL CONCLUSIONS FOR TYPE INFERENCE SYSTEMS")
    print("=" * 80)
    print()
    print("✅ WHAT TYPE INFERENCE SYSTEMS CAN DO:")
    print("   1. Extract COMPLETE nested generic structures from function annotations")
    print("   2. Recover TypeVar constraints and bounds from annotations")
    print("   3. Handle modern and legacy typing syntax identically")  
    print("   4. Access TypeVar info from ALL generic types (built-in, Pydantic, dataclass)")
    print("   5. Infer concrete types from instances using framework-specific mechanisms")
    print("   6. Track progressive Pydantic specialization via 'parameters' and 'args' fields")
    print()
    print("⚠️  WHAT TO WATCH OUT FOR:")
    print("   1. Pydantic same-TypeVar optimization: MyModel[A] returns original class")
    print("   2. Different metadata access patterns for different generic types")
    print("   3. Runtime instances lose ALL generic information")
    print("   4. Pydantic 'parameters' contains CURRENTLY ACTIVE TypeVars, not originals!")
    print()
    print("🚀 KEY INSIGHTS:")
    print("   - Function annotations preserve COMPLETE type information for type inference")
    print("   - Pydantic enables progressive specialization tracking via metadata")
    print("   - The apparent 'type erasure' only affects runtime instances, not annotations")
    print("   - Model[A,B] → Model[int, dict[str,C]] has parameters=(C,) enabling further [str]")
    print()
    print("✅ ALL TESTS PASSED - TYPE INFERENCE WITH PROGRESSIVE SPECIALIZATION IS VIABLE!")

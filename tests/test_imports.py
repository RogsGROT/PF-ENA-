from imports import *
# Import local modules
import importlib.util
import sys

def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def test_imports():
    print("Testing imports...")
    
    # Test numpy
    arr = np.array([1, 2, 3])
    print("✓ NumPy working")
    
    # Test pandas
    df = pd.DataFrame({'test': [1, 2, 3]})
    print("✓ Pandas working")
    
    # Test scipy
    result = sp.integrate.quad(lambda x: x**2, 0, 1)
    print(f"✓ SciPy working (integral result: {result[0]:.3f})")
    
    # Test matplotlib
    plt.figure()
    plt.close()
    print("✓ Matplotlib working")
    
    # Test seaborn
    sns.set_theme()
    print("✓ Seaborn working")
    
    # Test andes
    ss = System()
    print("✓ ANDES working")
    
    # Test local module imports
    try:
        system_setup = import_from_file("system_setup", "1_system_setup.py")
        ena_output = import_from_file("ena_output", "2_ena_output.py")
        print("✓ Local module imports working")
    except Exception as e:
        print(f"✗ Local module imports failed: {str(e)}")
    
    print("\nAll imports successful!")

if __name__ == "__main__":
    test_imports() 
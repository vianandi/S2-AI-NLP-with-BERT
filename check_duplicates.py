def check_duplicates_in_file(filepath="evaluate_advanced.py"):
    """Check for duplicate method definitions"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        methods = {}
        duplicates_found = []
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('def ') and ':' in line:
                # Extract method name
                method_name = stripped.split('def ')[1].split('(')[0].strip()
                
                if method_name in methods:
                    duplicates_found.append({
                        'method': method_name,
                        'first_line': methods[method_name],
                        'duplicate_line': i,
                        'content': stripped
                    })
                else:
                    methods[method_name] = i
        
        print(f"📋 Analysis of {filepath}:")
        print(f"Total methods found: {len(methods)}")
        print(f"Method names: {sorted(methods.keys())}")
        
        if duplicates_found:
            print(f"\n🔴 DUPLICATES FOUND ({len(duplicates_found)}):")
            for dup in duplicates_found:
                print(f"   Method: '{dup['method']}'")
                print(f"   First occurrence: Line {dup['first_line']}")
                print(f"   Duplicate at: Line {dup['duplicate_line']}")
                print(f"   Content: {dup['content']}")
                print("-" * 50)
            return False
        else:
            print("✅ No duplicate method definitions found!")
            return True
            
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return False
    except Exception as e:
        print(f"❌ Error checking file: {str(e)}")
        return False

def check_all_files():
    """Check all Python files for duplicates"""
    files_to_check = [
        "evaluate_advanced.py",
        "run_evaluation.py", 
        "models/model_factory.py",
        "models/advanced_ensemble.py"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"\n{'='*60}")
            check_duplicates_in_file(file)
        else:
            print(f"⚠️  File not found: {file}")

if __name__ == "__main__":
    import os
    check_all_files()
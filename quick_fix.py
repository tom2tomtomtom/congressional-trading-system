#!/usr/bin/env python3
"""
Quick Fix for Congressional Trading Dashboard - Add Missing Braces
"""

def quick_fix_dashboard():
    dashboard_path = "/Users/thomasdowuona-hyde/congressional-trading-system/src/dashboard/comprehensive_dashboard.html"
    
    print("🔧 Quick-fixing dashboard JavaScript syntax...")
    
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # List of function names that need braces fixed
    functions_to_fix = [
        'initializeMemberCharts',
        'initializeTradeCharts',
        'initializePatternCharts', 
        'initializePredictionCharts',
        'initializeCorrelationCharts',
        'initializeCommitteeCharts',
        'initializeLegislationCharts',
        'initializeLobbyingCharts',
        'initializeStockWatchlistCharts'
    ]
    
    fixes_applied = []
    
    for func_name in functions_to_fix:
        # Look for function declarations missing opening brace
        old_pattern = f"function {func_name}()\n"
        new_pattern = f"function {func_name}() {{\n"
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            fixes_applied.append(f"Added {{ to {func_name}")
    
    # Also fix any cases where there might be extra whitespace
    for func_name in functions_to_fix:
        old_pattern = f"function {func_name}() \n"
        new_pattern = f"function {func_name}() {{\n"
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            fixes_applied.append(f"Added {{ to {func_name} (whitespace variant)")
    
    # Save the fixed version
    fixed_path = dashboard_path.replace('.html', '_WORKING.html')
    with open(fixed_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed dashboard saved as: {fixed_path}")
    print(f"🔧 Applied {len(fixes_applied)} fixes:")
    for fix in fixes_applied:
        print(f"   • {fix}")
    
    return fixed_path

if __name__ == "__main__":
    fixed_file = quick_fix_dashboard()
    print(f"\n🚀 Opening fixed dashboard: {fixed_file}")
    
    import subprocess
    subprocess.run(['open', fixed_file])

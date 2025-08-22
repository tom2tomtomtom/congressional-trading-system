# Clean Codebase Workflow

<workflow_meta>
  <name>clean-codebase</name>
  <description>Automated code formatting and cleanup for Congressional Trading Intelligence System</description>
  <estimated_time>3-5 minutes</estimated_time>
  <frequency>before commits or when code quality issues detected</frequency>
</workflow_meta>

## Workflow Steps

<workflow_steps>
  <step number="1">
    <command>@python-formatting</command>
    <description>Apply Python code formatting standards</description>
    <condition>always</condition>
    <tools>black, isort</tools>
    <script>
      cd /Users/thomasdowuona-hyde/congressional-trading-system
      python3 -m black src/ --line-length 88 --target-version py39
      python3 -m isort src/ --profile black
    </script>
  </step>
  
  <step number="2">
    <command>@python-linting</command>
    <description>Run Python linting and style checks</description>
    <condition>always</condition>
    <tools>flake8</tools>
    <script>
      cd /Users/thomasdowuona-hyde/congressional-trading-system
      python3 -m flake8 src/ --max-line-length=88 --extend-ignore=E203,W503
    </script>
    <expected_output>No linting errors or warnings</expected_output>
  </step>
  
  <step number="3">
    <command>@javascript-formatting</command>
    <description>Format JavaScript and CSS files</description>
    <condition>if_frontend_changes</condition>
    <focus_areas>
      - src/dashboard/static/js/
      - src/dashboard/static/css/
      - React components (if present)
    </focus_areas>
    <actions>
      - Remove console.log statements
      - Fix indentation consistency
      - Remove commented code blocks
      - Validate HTML structure
    </actions>
  </step>
  
  <step number="4">
    <command>@remove-debug-code</command>
    <description>Clean up debug statements and test code</description>
    <condition>always</condition>
    <cleanup_targets>
      - print() statements in production code
      - Debug logging above INFO level
      - Commented-out code blocks
      - Temporary test files
      - Unused import statements
    </cleanup_targets>
  </step>
  
  <step number="5">
    <command>@validate-imports</command>
    <description>Organize and validate import statements</description>
    <condition>always</condition>
    <actions>
      - Remove unused imports
      - Sort imports by standard library, third-party, local
      - Check for circular dependencies
      - Validate all imports resolve correctly
    </actions>
  </step>
  
  <step number="6">
    <command>@documentation-cleanup</command>
    <description>Update and format documentation</description>
    <condition>if_documentation_present</condition>
    <focus_areas>
      - Docstring formatting and completeness
      - README file accuracy
      - Inline comments cleanup
      - API documentation consistency
    </focus_areas>
  </step>
  
  <step number="7">
    <command>@file-organization</command>
    <description>Organize files and remove cruft</description>
    <condition>always</condition>
    <cleanup_actions>
      - Remove __pycache__ directories
      - Clean .pyc files
      - Remove temporary log files
      - Organize data files properly
      - Clean up test output files
    </cleanup_actions>
  </step>
</workflow_steps>

## Code Quality Standards

### Python Standards
- **Line Length**: 88 characters (Black default)
- **Import Organization**: stdlib, third-party, local (isort)
- **String Quotes**: Double quotes for strings, single for code
- **Docstrings**: Google style for functions and classes

### JavaScript Standards
- **Indentation**: 2 spaces for JS/CSS, 4 spaces for HTML
- **Semicolons**: Required for all statements
- **Quote Style**: Single quotes for strings
- **Variable Naming**: camelCase for JS, kebab-case for CSS

### General Standards
- **File Encoding**: UTF-8 for all text files
- **Line Endings**: LF (Unix style)
- **Trailing Whitespace**: Remove all trailing spaces
- **Final Newline**: Ensure files end with newline

## Automated Fixes

<fix_categories>
  <category name="formatting">
    <description>Automatic code formatting fixes</description>
    <fixes>
      - Line length adjustment
      - Indentation normalization
      - Quote style consistency
      - Import statement organization
    </fixes>
  </category>
  
  <category name="cleanup">
    <description>Remove unwanted code and files</description>
    <fixes>
      - Debug print statements
      - Commented code blocks
      - Unused import statements
      - Temporary files and directories
    </fixes>
  </category>
  
  <category name="structure">
    <description>File and directory organization</description>
    <fixes>
      - Move misplaced files to correct directories
      - Rename files to match conventions
      - Create missing __init__.py files
      - Remove empty directories
    </fixes>
  </category>
</fix_categories>

## Quality Gates

### Before Cleanup
- [ ] Backup current working state
- [ ] Verify no critical processes running
- [ ] Check git status for uncommitted changes

### During Cleanup
- [ ] Track all automated changes made
- [ ] Preserve functional code behavior
- [ ] Maintain file permissions and ownership
- [ ] Handle file conflicts gracefully

### After Cleanup
- [ ] Run basic functionality tests
- [ ] Verify no imports broken
- [ ] Check dashboard still loads
- [ ] Validate analysis engine runs

## Error Handling

<error_scenarios>
  <scenario name="formatting_tool_missing">
    <condition>black, isort, or flake8 not installed</condition>
    <action>
      1. Check pyproject.toml dev dependencies
      2. Suggest pip install commands
      3. Provide manual formatting guidelines
    </action>
  </scenario>
  
  <scenario name="syntax_errors_detected">
    <condition>Code has syntax errors preventing formatting</condition>
    <action>
      1. Identify specific syntax errors
      2. Report location and nature of errors
      3. Skip automated formatting for problematic files
      4. Provide manual fix recommendations
    </action>
  </scenario>
  
  <scenario name="import_resolution_failures">
    <condition>Imports cannot be resolved or organized</condition>
    <action>
      1. List unresolved imports
      2. Check if dependencies are installed
      3. Verify virtual environment activation
      4. Suggest dependency installation
    </action>
  </scenario>
</error_scenarios>

## Integration Points

**Pre-Commit Hook**: Run automatically before git commits
**IDE Integration**: Compatible with VS Code, PyCharm formatting
**CI/CD Pipeline**: Include in automated testing workflow
**Daily Workflow**: Part of routine development maintenance

## Success Criteria

- [ ] All Python files properly formatted with Black
- [ ] No flake8 linting errors or warnings
- [ ] Import statements correctly organized
- [ ] Debug code and commented blocks removed
- [ ] File structure clean and organized
- [ ] Documentation updated and consistent
- [ ] No functionality broken by cleanup process